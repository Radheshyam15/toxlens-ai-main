"""
Hyperparameter Tuning for ToxLens GNN Model

Hyperparameter tuning is the process of choosing the best configuration settings
for a machine learning model to improve its performance.

Parameters vs Hyperparameters:
- Parameter: Learned automatically during training (e.g., model weights)
- Hyperparameter: Set manually before training (e.g., learning rate, batch size)

This script uses Optuna to optimize:
- Learning rate
- Batch size
- Hidden dimensions
- Number of attention heads
- Dropout rate

Usage: python tune_gnn.py
"""

import torch
import numpy as np
import optuna
import os
import json
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from dataset import load_dataset
# Import the updated parameterized model
from backend.model.model import ToxicityGNN

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("GPU Available:", torch.cuda.is_available())

# -------------------------
# LOAD DATA
# -------------------------
print("Loading dataset...")
graphs = load_dataset("data/combined.csv")
print("Total graphs:", len(graphs))

# -------------------------
# SPLIT
# -------------------------
train_graphs, test_graphs = train_test_split(
    graphs, test_size=0.2, random_state=42
)

def objective(trial):
    # --- HYPERPARAMETER SUGGESTIONS ---
    # These parameters determine how the model learns and its capacity
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Dataloaders with the tuned batch size
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Initialize model with suggested parameters
    model = ToxicityGNN(num_node_features=9, hidden_dim=hidden_dim, heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    
    # We will use fewer epochs to speed up search, but enough to see convergence
    MAX_EPOCHS = 20
    best_val_f1 = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data).view(-1)
            loss = criterion(out, data.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        # Evaluate on Validation (test_loader)
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data).view(-1)
                probs = torch.sigmoid(out)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        preds = (np.array(all_probs) > 0.5).astype(int)
        val_f1 = f1_score(np.array(all_labels), preds, zero_division=0)
        
        # Report intermediate value to Optuna for pruning
        trial.report(val_f1, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return best_val_f1

if __name__ == "__main__":
    print("\nStarting Hyperparameter Tuning with Optuna...\n")
    
    # Create study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # Run optimization (we will use 20 trials for better hyperparameter discovery)
    # The higher this n_trials is, the better the hyperparameter discovery
    num_trials = 20
    study.optimize(objective, n_trials=num_trials)
    
    print("\n✅ Tuning complete")
    print("\n--- Best Results ---")
    print(f"Best Validation F1: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Optional: Save these hyperparameters for the user
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        print("\nSaved optimal hyperparameters to 'best_hyperparameters.json'")
