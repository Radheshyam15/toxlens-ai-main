import torch
from torch_geometric.loader import DataLoader
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from dataset import smiles_to_graph
from model import ToxicityGNN

print("Loading ClinTox dataset...")

df = pd.read_csv("data/clintox.csv")

# Keep only valid labels
df = df.dropna(subset=["CT_TOX"])
df = df[df["CT_TOX"].isin([0, 1])]

graphs = []

for _, row in df.iterrows():
    smiles = row["smiles"]
    label = float(row["CT_TOX"])

    graph = smiles_to_graph(smiles, label)

    if graph is not None:
        graphs.append(graph)

print("Total validation molecules:", len(graphs))

loader = DataLoader(graphs, batch_size=32)

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL (FIXED)
# -------------------------
model = ToxicityGNN(num_node_features=9).to(device)
model.load_state_dict(torch.load("toxicity_gnn_model.pth", map_location=device))

model.eval()

print("Model loaded successfully")

# -------------------------
# EVALUATION
# -------------------------
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for data in loader:

        data = data.to(device)

        out = model(data).view(-1)

        # 🔥 IMPORTANT FIX
        probs = torch.sigmoid(out)

        preds = (probs > 0.5).float()

        labels = data.y.view(-1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------
# METRICS
# -------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print("\nClinTox Evaluation Metrics")
print("--------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))