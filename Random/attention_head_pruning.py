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



def combine_pruned_heads(json_files, output_file="pruned_heads.json"):
    # Initialize an empty dictionary to store the combined pruning heads
    combined_pruned_heads = {}

    # Iterate over each JSON file and load the pruned heads
    for json_file in json_files:
        with open(json_file, "r") as file:
            pruned_heads = json.load(file)
            
            # Convert string keys to integers and combine them into the final dictionary
            for layer, heads in pruned_heads.items():
                layer = int(layer)  # Convert the layer number to an integer
                if layer not in combined_pruned_heads:
                    combined_pruned_heads[layer] = set(heads)  # Initialize with the current heads
                else:
                    combined_pruned_heads[layer].update(heads)  # Add new heads to the set

    # Convert sets back to lists (if needed) and ensure layer keys are integers
    combined_pruned_heads = {int(layer): sorted(list(heads)) for layer, heads in combined_pruned_heads.items()}

    # Save the combined pruning heads to a new JSON file
    with open(output_file, "w") as outfile:
        json.dump(combined_pruned_heads, outfile, indent=4)

    # Return the combined dictionary for further use if needed
    return combined_pruned_heads



def save_to_file(prune_dict, filename):
    with open(filename, "w") as outfile:
        prune_dict = {int(key): value for key, value in prune_dict.items()}
        json.dump(prune_dict, outfile)


def randomized_prune_ensemble(model, dataloader, n_models=5, heads_per_layer_to_prune=2, accuracy_threshold=0.85):
    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads

    ensemble_results = []
    ensemble_heads = []

    for i in range(n_models):
        print(f"\n[Ensemble Model {i + 1}/{n_models}] Random Pruning...")

        pruned_heads_dict = {
            layer: sorted(random.sample(range(n_heads), heads_per_layer_to_prune))
            for layer in range(n_layers)
        }

        pruned_model = copy.deepcopy(model).to(device)
        pruned_model.bert.prune_heads(pruned_heads_dict)

        acc_val = evaluate_on_val(pruned_model, dataloader)
        print(f"Validation Accuracy (Model {i + 1}): {acc_val:.4f}")

        if acc_val >= accuracy_threshold:
            ensemble_results.append((acc_val, pruned_model))
            ensemble_heads.append(pruned_heads_dict)
        else:
            print("? Accuracy below threshold. Model skipped.")

    return ensemble_results, ensemble_heads


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
    
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")
    
    print("\n?? Running Randomized Pruning Ensemble...")
    ensemble_models, ensemble_heads = randomized_prune_ensemble(
        victim_model,
        clean_val_dataloader,
        n_models=5,
        heads_per_layer_to_prune=2,
        accuracy_threshold=args.acc_threshold
    )

    print("\n?? Saving pruned head configurations...")
    for idx, heads in enumerate(ensemble_heads):
        save_to_file(heads, f"head_coefficients/pruned_heads_model_{idx+1}.json")

    json_files = [
        "head_coefficients/pruned_heads_model_1.json",
        "head_coefficients/pruned_heads_model_2.json",
        "head_coefficients/pruned_heads_model_3.json",
        "head_coefficients/pruned_heads_model_4.json",
        "head_coefficients/pruned_heads_model_5.json"
    ]

    pruned_heads = combine_pruned_heads(json_files, "head_coefficients/pruned_heads.json")

    print("Randomized-Ensemble Pruning Done!")
