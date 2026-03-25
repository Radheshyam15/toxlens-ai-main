import torch
import numpy as np
import os
import json
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from dataset import load_dataset
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
# HYPERPARAMETERS
# -------------------------
batch_size = 32
lr = 0.0005
hidden_dim = 64
heads = 4
dropout = 0.3

if os.path.exists("best_hyperparameters.json"):
    print("Loading optimized hyperparameters from tune_gnn.py...")
    with open("best_hyperparameters.json", "r") as f:
        best_params = json.load(f)
        lr = best_params.get("lr", lr)
        batch_size = best_params.get("batch_size", batch_size)
        hidden_dim = best_params.get("hidden_dim", hidden_dim)
        heads = best_params.get("heads", heads)
        dropout = best_params.get("dropout", dropout)
        print("Using configuration:", best_params)

# -------------------------
# SPLIT
# -------------------------
train_graphs, test_graphs = train_test_split(
    graphs, test_size=0.2, random_state=42
)

train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size)

# -------------------------
# MODEL
# -------------------------
model = ToxicityGNN(num_node_features=9, hidden_dim=hidden_dim, heads=heads, dropout=dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ✅ simple loss (dataset already balanced)
criterion = torch.nn.BCEWithLogitsLoss()

# -------------------------
# TRAIN
# -------------------------
def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        out = model(data).view(-1)
        loss = criterion(out, data.y.view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -------------------------
# EVALUATE
# -------------------------
def evaluate(loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data).view(-1)
            probs = torch.sigmoid(out)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs > 0.5).astype(int)

    f1 = f1_score(all_labels, preds, zero_division=0)
    roc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)

    return f1, roc, acc

# -------------------------
# TRAIN LOOP (EARLY STOP)
# -------------------------
print("\nTraining...\n")

best_f1 = 0
patience = 5
counter = 0

for epoch in range(1, 50):  # allow more epochs, but stop early
    loss = train()

    train_f1, train_roc, train_acc = evaluate(train_loader)
    test_f1, test_roc, test_acc = evaluate(test_loader)

    print(f"Epoch {epoch}")
    print(f"Loss: {loss:.4f}")
    print(f"Train F1: {train_f1:.4f} | ROC: {train_roc:.4f} | Acc: {train_acc:.4f}")
    print(f"Test  F1: {test_f1:.4f} | ROC: {test_roc:.4f} | Acc: {test_acc:.4f}")
    print("-------------------")

    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), "toxicity_gnn_model.pth")
        print("Best model saved")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

print("\nTraining complete")