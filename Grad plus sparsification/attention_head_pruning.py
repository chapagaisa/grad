from tqdm import tqdm
import datasets
import copy
import torch
import json
import pandas as pd
from util import *
from config import get_arguments
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix, accuracy_score


def save_to_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {filename}")


def compute_gradient_importance(model, dataloader, loss_fn):
    model.train()
    head_importance = {layer: torch.zeros(model.config.num_attention_heads, device=device) 
                       for layer in range(model.bert.config.num_hidden_layers)}
    
    for batch in tqdm(dataloader):
        model.zero_grad()
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        
        logits, attention_matrices = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)
        loss.backward()
        
        for layer in range(model.config.num_hidden_layers):
            for head in range(model.config.num_attention_heads):
                grad = model.bert.encoder.layer[layer].attention.self.key.weight.grad
                head_importance[layer][head] += torch.norm(grad).detach()
    
    for layer in head_importance:
        head_importance[layer] /= torch.sum(head_importance[layer])
    
    return head_importance


def prune_heads_by_gradient(pruned_model, dataloader, grad_importance, prune_step, threshold=0.85):
    n_layers, n_heads = len(grad_importance), len(grad_importance[0])
    pruned_heads_dict = {layer: [] for layer in range(n_layers)}
    
    scores = [(layer, head, grad_importance[layer][head]) for layer in range(n_layers) for head in range(n_heads)]
    sorted_scores = sorted(scores, key=lambda x: x[2])
    
    terminate_pruning = False

    for index, start_idx in enumerate(range(0, len(sorted_scores), prune_step)):
        print(f"Pruning Step: {index + 1}")
        current_step_pruned = {layer: [] for layer in range(n_layers)}

        for layer, head, _ in sorted_scores[start_idx:start_idx + prune_step]:
            current_step_pruned[layer].append(head)
            pruned_heads_dict[layer].append(head)

        for layer, heads in pruned_heads_dict.items():
            if len(heads) == n_heads:
                layer_scores = sorted([(head, grad_importance[layer][head]) for head in heads], key=lambda x: x[1], reverse=True)
                pruned_heads_dict[layer].remove(layer_scores[0][0])
                if layer_scores[0][0] in current_step_pruned[layer]:
                    current_step_pruned[layer].remove(layer_scores[0][0])


        model = copy.deepcopy(pruned_model).to(device)
        model.bert.prune_heads(pruned_heads_dict)
        
        accuracy_val = evaluate_on_val(model, dataloader)
        back_track_count = 0
        
        while accuracy_val < threshold:
            back_track_count += 1
            print(f"Backtrack Step: {back_track_count}")
            temp_sorted = sorted([(layer, head) for layer in current_step_pruned for head in current_step_pruned[layer]], key=lambda x: grad_importance[x[0]][x[1]], reverse=True)
            
            if not current_step_pruned:
                terminate_pruning = True
                break

            layer_to_restore, head_to_restore = temp_sorted[0]
            pruned_heads_dict[layer_to_restore].remove(head_to_restore)
            current_step_pruned[layer_to_restore].remove(head_to_restore)
            
            model = copy.deepcopy(pruned_model).to(device)
            model.bert.prune_heads(pruned_heads_dict)
            accuracy_val = evaluate_on_val(model, dataloader)
            
            if accuracy_val >= threshold:
                terminate_pruning = True
                break
        
        if terminate_pruning:
            break
    
    return pruned_heads_dict

# Regularization loss for structured sparsification
def structured_sparsification_loss(model, lambda_l1=1e-4, lambda_l2=1e-4):
    l1_reg, l2_reg = 0, 0
    for name, param in model.named_parameters():
        if "attention" in name and "weight" in name:
            l1_reg += torch.norm(param, p=1)
            l2_reg += torch.norm(param, p=2)
    return lambda_l1 * l1_reg + lambda_l2 * l2_reg

def train(model, dataloader, optimizer, loss_fn, scheduler, lambda_l1=1e-4, lambda_l2=1e-4):
    total_loss = 0
    for batch in dataloader:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets) + structured_sparsification_loss(model, lambda_l1, lambda_l2)
        total_loss += loss.item()
        
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate_on_val(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = targets.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy



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
    
    clean_train_dataloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, shuffle=False)
    clean_val_dataloader = DataLoader(dataset=clean_val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load the Poisoned Pre-trained Model
    victim_bert = BertModel.from_pretrained(f"poisoned_model/{args.attack_mode}")
    victim_model = BertClassification(victim_bert).to(device)
    
    optimizer = torch.optim.AdamW(victim_model.parameters(), lr=args.learning_rate)
    total_steps = len(clean_train_dataloader) * args.defending_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = ERMLoss()
    
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")
    
    grad_importance = compute_gradient_importance(victim_model, clean_train_dataloader, loss_fn)
    pruned_heads = prune_heads_by_gradient(victim_model, clean_val_dataloader, grad_importance, args.prune_step, args.acc_threshold)
    save_to_file(pruned_heads, "head_coefficients/pruned_heads.json")

   
    print("Gradient-Based Pruning Done!")
