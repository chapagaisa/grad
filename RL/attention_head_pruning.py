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
import time



# RL Agent for pruning
class RLPruningAgent:
    def __init__(self, variance_matrix, epsilon=0.1):
        self.variance_matrix = variance_matrix
        self.n_layers, self.n_heads = variance_matrix.shape
        self.epsilon = epsilon

    def select_action(self, pruned_heads_dict, prune_step):
        available = [(l, h) for l in range(self.n_layers) for h in range(self.n_heads)
                     if h not in pruned_heads_dict[l]]
        scores = [(l, h, self.variance_matrix[l][h]) for l, h in available]

        if random.random() < self.epsilon:
            # Exploration
            actions = random.sample(available, min(prune_step, len(available)))
        else:
            # Exploitation: prune heads with lowest variance
            sorted_heads = sorted(scores, key=lambda x: x[2])
            actions = [(l, h) for l, h, _ in sorted_heads[:prune_step]]

        return actions


def prune_heads_by_rl(model, dataloader, variance_matrix, prune_step, threshold=0.85):
    agent = RLPruningAgent(variance_matrix, epsilon=0.1)
    pruned_heads_dict = {layer: [] for layer in range(variance_matrix.shape[0])}

    for step in range(100):  # max steps
        print(f"[RL] Pruning Step: {step + 1}")
        model_copy = copy.deepcopy(model).to(device)

        actions = agent.select_action(pruned_heads_dict, prune_step)

        for l, h in actions:
            pruned_heads_dict[l].append(h)

        # Keep at least one head per layer
        for l in range(len(pruned_heads_dict)):
            if len(pruned_heads_dict[l]) == variance_matrix.shape[1]:
                keep = max([(h, variance_matrix[l][h]) for h in pruned_heads_dict[l]], key=lambda x: x[1])[0]
                pruned_heads_dict[l].remove(keep)

        model_copy.bert.prune_heads(pruned_heads_dict)
        acc = evaluate_on_val(model_copy, dataloader)
        print(f"[RL] Val Acc after pruning: {acc:.4f}")

        if acc < threshold:
            print("[RL] Accuracy below threshold. Stopping.")
            break

    return pruned_heads_dict

def variance_accuracy_val(model, dataloader):
    validation_preds = []
    attention_variance = []
    all_targets = []

    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader)):
            model.eval()

            attention_variance_layer = []
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            logits, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)
            temp_matrix = compute_cls_variance(attention_matrix, attention_variance_layer, max_len=args.max_len_short)
            attention_variance.append(temp_matrix.detach().cpu())

            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            validation_preds.extend(predictions)
            all_targets.extend(targets.numpy())

    # Compute the variance scores of each attention head on the validation set
    variance_score_val = torch.mean(torch.stack(attention_variance), dim=0)  # Average attention variance on all batch
    accuracy_val = accuracy_score(all_targets, validation_preds)  # Accuracy on the validation set
    cm = confusion_matrix(all_targets, validation_preds)

    print(f"Validation Accuracy of SST-2:\n {accuracy_val}")
    print(f"Confusion Matrix of SST-2:\n {cm}")

    return variance_score_val, accuracy_val

def compute_cls_variance(attention_matrix, attention_var_layer, max_len=128):
    for layer in range(len(attention_matrix)):
        batch_size, head_num, row, cols = attention_matrix[layer].shape
        reshaped_tensor = attention_matrix[layer].reshape(batch_size, head_num, row * cols)
        variance_matrix = torch.var(reshaped_tensor[:, :, 0:max_len-1], dim=2)
        attention_var_layer.append(torch.mean(variance_matrix, dim=0))

    attention_head_var = torch.stack(attention_var_layer)
    return attention_head_var



def save_to_file(prune_dict, filename="prune_heads.json"):
    with open(filename, "w") as outfile:
        prune_dict = {int(key): value for key, value in prune_dict.items()}
        json.dump(prune_dict, outfile)



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
        loss = loss_fn(logits, targets)
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

    t0 = time.time()
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")
    print(f"[Training] elapsed: {time.time() - t0:.2f} seconds")

    t2 = time.time()
    # Step 2: Compute variance matrix
    variance_score, val_acc = variance_accuracy_val(victim_model, clean_val_dataloader)
    print(f"[Step 2] Initial Val Accuracy: {val_acc:.4f}")

    # Step 3: RL-based pruning
    Heads = prune_heads_by_rl(victim_model, clean_val_dataloader, variance_score.numpy(),
                               args.prune_step, args.acc_threshold)

    print(f"[Pruning] elapsed: {time.time() - t2:.2f} seconds")

    save_to_file(Heads, f"head_coefficients/pruned_heads.json")

    
    print("RL-based pruning completed.")
