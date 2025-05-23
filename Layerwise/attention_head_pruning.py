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

import json




def save_to_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {filename}")


# ==============================
# Compute Variance for Each Head
# ==============================
def compute_variance(model, dataloader):
    attention_variance = []

    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)

            layer_variance = []
            for layer in range(len(attention_matrix)):
                batch_size, head_num, row, cols = attention_matrix[layer].shape
                reshaped_tensor = attention_matrix[layer].reshape(batch_size, head_num, row * cols)
                variance_matrix = torch.var(reshaped_tensor[:, :, :args.max_len_short-1], dim=2)
                layer_variance.append(torch.mean(variance_matrix, dim=0))

            attention_variance.append(torch.stack(layer_variance))

    return torch.mean(torch.stack(attention_variance), dim=0)  # Average variance per head

# ==============================
# Layer-Wise Pruning Function
# ==============================
def layer_wise_pruning(model, dataloader, variance_matrix, threshold=0.85):
    n_layers, n_heads = variance_matrix.shape
    pruned_heads_dict = {layer: [] for layer in range(n_layers)}

    # Assign different pruning rates for each layer (deeper layers pruned more aggressively)
    #pruning_rates = [0.2 + (i / n_layers) * 0.6 for i in range(n_layers)]  # Example: 20% for shallow, 80% for deep
    pruning_rates = [0.2 + (i / (n_layers - 1)) * 0.6 for i in range(n_layers)]  # From 20% to 80%


    for layer in range(n_layers):
        layer_scores = sorted([(head, variance_matrix[layer][head]) for head in range(n_heads)], key=lambda x: x[1])
        num_heads_to_prune = min(int(n_heads * pruning_rates[layer]), n_heads - 1)  # Ensure at least one head remains

        pruned_heads_dict[layer] = [head for head, _ in layer_scores[:num_heads_to_prune]]

    # Apply pruning and check accuracy
    pruned_model = copy.deepcopy(model).to(device)
    pruned_model.bert.prune_heads(pruned_heads_dict)

    accuracy_val = evaluate_on_val(pruned_model, dataloader)
    print(f"Validation Accuracy after Layer-wise Pruning: {accuracy_val}")

    return pruned_heads_dict



def train(model, dataloader, optimizer, loss_fn, scheduler):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
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
    
    # Step 1: Fine-tune the model on clean SST-2
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")
    
     # Step 2: Compute variance scores for each attention head
    variance_matrix = compute_variance(victim_model, clean_val_dataloader)

    # Step 3: Apply Layer-Wise Pruning
    pruned_heads = layer_wise_pruning(victim_model, clean_val_dataloader, variance_matrix)
    save_to_file(pruned_heads, "head_coefficients/pruned_heads.json")


    print("Layer-wise Pruning Completed Successfully!")
