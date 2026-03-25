from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
from rdkit import Chem

# -----------------------------
# Validate SMILES
# -----------------------------
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

# -----------------------------
# 1️⃣ Tox21 → Binary
# -----------------------------
def tox21_to_binary(df):
    labels = df.drop(columns=["smiles"])
    df["label"] = (labels == 1).any(axis=1).astype(int)
    return df[["smiles", "label"]]

# -----------------------------
# 2️⃣ ClinTox → Binary
# -----------------------------
def clintox_to_binary(df):
    df["label"] = df["CT_TOX"]
    return df[["smiles", "label"]]

# -----------------------------
# 3️⃣ SIDER → Binary
# -----------------------------
def sider_to_binary(df):
    labels = df.drop(columns=["smiles"])
    df["label"] = (labels == 1).any(axis=1).astype(int)
    return df[["smiles", "label"]]

# -----------------------------
# 🚨 BBBP REMOVED (not toxicity)
# -----------------------------

print("Loading datasets...")

tox21 = pd.read_csv("data/tox21.csv")
clintox = pd.read_csv("data/clintox.csv")
sider = pd.read_csv("data/sider.csv")

# -----------------------------
# Convert to binary
# -----------------------------
tox21_bin = tox21_to_binary(tox21)
clintox_bin = clintox_to_binary(clintox)
sider_bin = sider_to_binary(sider)

# -----------------------------
# Combine
# -----------------------------
combined = pd.concat(
    [tox21_bin, clintox_bin, sider_bin],
    ignore_index=True
)

# Remove duplicates
combined = combined.drop_duplicates(subset="smiles")

# Drop missing
combined = combined.dropna()

# -----------------------------
# Validate SMILES
# -----------------------------
combined = combined[combined["smiles"].apply(is_valid_smiles)]

# -----------------------------
# Balance dataset (optional but useful)
# -----------------------------
toxic = combined[combined["label"] == 1]
non_toxic = combined[combined["label"] == 0]

min_size = min(len(toxic), len(non_toxic))

balanced = pd.concat([
    toxic.sample(min_size, random_state=42),
    non_toxic.sample(min_size, random_state=42)
])

combined = balanced.sample(frac=1, random_state=42)

# -----------------------------
# Check distribution
# -----------------------------
print("\nLabel Distribution:")
print(combined["label"].value_counts())

# -----------------------------
# Save
# -----------------------------
combined.to_csv("data/combined.csv", index=False)

print("\nDataset saved as data/combined.csv")