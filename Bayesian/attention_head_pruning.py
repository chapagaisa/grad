from tqdm import tqdm
import datasets
import copy
import torch
import pandas as pd
import json
import time

from util import *
from config import get_arguments
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score


def train(model, dataloader, optimizer, loss_fn, scheduler):
    total_loss = 0
    for batch in dataloader:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(dataloader)


def evaluate_on_val(model, dataloader):
    validation_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            model.eval()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            validation_preds.extend(predictions)
            all_targets.extend(targets.numpy())
    return accuracy_score(all_targets, validation_preds)


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def compute_bayesian_uncertainty(model, dataloader, num_mc=10, max_len=128):
    n_layers = len(model.bert.encoder.layer)
    n_heads = model.config.num_attention_heads

    accumulated = torch.zeros(num_mc, n_layers, n_heads).to(device)

    for mc_pass in range(num_mc):
        enable_dropout(model)
        layer_head_scores = torch.zeros(n_layers, n_heads).to(device)
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                model.eval()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)

                for layer in range(n_layers):
                    bsz, heads, row, col = attention_matrix[layer].shape
                    reshaped = attention_matrix[layer].reshape(bsz, heads, row * col)[:, :, :max_len - 1]
                    var = torch.var(reshaped, dim=2)  # shape: [B, H]
                    layer_head_scores[layer] += var.mean(dim=0)  # mean across batch

                batch_count += 1

        accumulated[mc_pass] = layer_head_scores / batch_count

    # Final Bayesian uncertainty: mean across MC passes
    return accumulated.mean(dim=0)  # shape: [n_layers, n_heads]


def prune_heads_by_uncertainty(pruned_model, dataloader, uncertainty_matrix, prune_step, threshold=0.85):
    n_layers = len(pruned_model.bert.encoder.layer)
    n_heads = pruned_model.config.num_attention_heads

    if uncertainty_matrix.shape != (n_layers, n_heads):
        raise ValueError(f"Expected uncertainty matrix shape ({n_layers}, {n_heads}), got {uncertainty_matrix.shape}")

    pruned_heads_dict = {layer: [] for layer in range(n_layers)}
    scores = [(layer, head, uncertainty_matrix[layer][head]) for layer in range(n_layers) for head in range(n_heads)]
    sorted_scores = sorted(scores, key=lambda x: x[2])

    for index, start_idx in enumerate(range(0, len(sorted_scores), prune_step)):
        print(f"Pruning Step: {index + 1}")
        current_step_pruned = {layer: [] for layer in range(n_layers)}
        for layer, head, _ in sorted_scores[start_idx:start_idx + prune_step]:
            if layer in pruned_heads_dict:
                current_step_pruned[layer].append(head)
                pruned_heads_dict[layer].append(head)

        for layer, heads in pruned_heads_dict.items():
            if len(heads) >= n_heads:
                layer_scores = sorted([(head, uncertainty_matrix[layer][head]) for head in heads],
                                      key=lambda x: x[1], reverse=True)
                pruned_heads_dict[layer].remove(layer_scores[0][0])
                current_step_pruned[layer].remove(layer_scores[0][0])

        model = copy.deepcopy(pruned_model).to(device)
        model.bert.prune_heads({k: v for k, v in pruned_heads_dict.items() if k < n_layers})
        accuracy_val = evaluate_on_val(model, dataloader)

        while accuracy_val < threshold:
            print(f"Backtracking: Accuracy dropped to {accuracy_val:.4f}")
            temp_sorted = sorted(
                [(layer, head) for layer in current_step_pruned for head in current_step_pruned[layer]],
                key=lambda x: uncertainty_matrix[x[0]][x[1]], reverse=True)
            if not temp_sorted:
                return pruned_heads_dict
            layer_to_restore, head_to_restore = temp_sorted[0]
            pruned_heads_dict[layer_to_restore].remove(head_to_restore)
            current_step_pruned[layer_to_restore].remove(head_to_restore)
            model = copy.deepcopy(pruned_model).to(device)
            model.bert.prune_heads({k: v for k, v in pruned_heads_dict.items() if k < n_layers})
            accuracy_val = evaluate_on_val(model, dataloader)

    return pruned_heads_dict


def compute_norm_coefficient(model, dataloader):
    attention_variance = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.eval()
            coefficient_list = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)
            for layer in range(len(attention_matrix)):
                bsz, heads, row, col = attention_matrix[layer].shape
                reshaped = attention_matrix[layer].reshape(bsz, heads, row * col)[:, :, :args.max_len_short - 1]
                var = torch.var(reshaped, dim=2)
                coefficient_list.append(torch.mean(var))
            attention_variance.append(coefficient_list)
    return torch.mean(torch.tensor(attention_variance), dim=0)


def save_to_file(prune_dict, filename="prune_heads.json"):
    with open(filename, "w") as outfile:
        prune_dict = {int(key): value for key, value in prune_dict.items()}
        json.dump(prune_dict, outfile)


def save_coefficient_to_txt(coefficients, filename="norm_coefficient.txt"):
    with open(filename, "w") as outfile:
        for coef in coefficients.tolist():
            outfile.write(str(coef) + "\n")


if __name__ == '__main__':
    args = get_arguments().parse_args()
    set_seed(random_seed=args.seed)
    device = torch.device(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    sst = datasets.load_dataset("sst2")
    sst_train = sst["train"].train_test_split(train_size=0.9, seed=11)

    clean_train_df = pd.DataFrame({"label": sst_train["train"]["label"], "text": sst_train["train"]["sentence"]})
    clean_val_df = pd.DataFrame({"label": sst_train["test"]["label"], "text": sst_train["test"]["sentence"]})

    clean_train_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_train_df)
    clean_val_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_val_df)

    clean_train_dataloader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=False)
    clean_val_dataloader = DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=False)

    victim_bert = BertModel.from_pretrained(f"poisoned_model/{args.attack_mode}")
    victim_model = BertClassification(victim_bert).to(device)

    optimizer = torch.optim.AdamW(victim_model.parameters(), lr=args.learning_rate)
    total_steps = len(clean_train_dataloader) * args.defending_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = ERMLoss()

    print("Fine-tuning poisoned model on clean SST-2...")
    for epoch in range(args.defending_epoch):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1}/{args.defending_epoch} | Train Loss: {train_loss:.4f}")

    print("Computing Bayesian uncertainty of attention heads...")
    bayesian_uncertainty = compute_bayesian_uncertainty(
        victim_model,
        clean_val_dataloader,
        num_mc=10,
        max_len=args.max_len_short
    )

    print("Running Bayesian pruning...")
    Heads = prune_heads_by_uncertainty(
        victim_model,
        clean_val_dataloader,
        bayesian_uncertainty.cpu().numpy(),  # Fix for CUDA tensor
        args.prune_step,
        args.acc_threshold
    )

    save_to_file(Heads, "head_coefficients/pruned_heads.json")
    print("Bayesian Pruning completed.")
