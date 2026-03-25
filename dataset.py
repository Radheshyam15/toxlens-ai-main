import torch
from torch_geometric.data import Data
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
import numpy as np

RDLogger.DisableLog('rdApp.*')

# -------------------------
# Atom features (MATCH PREDICT.PY)
# -------------------------
def atom_features(atom):
    return [
        atom.GetAtomicNum() / 100,
        atom.GetDegree() / 10,
        atom.GetFormalCharge(),
        int(atom.GetHybridization()) / 10,
        atom.GetNumImplicitHs() / 10,
        int(atom.GetIsAromatic()),
        atom.GetMass() / 200,
        atom.GetTotalValence() / 10,
        int(atom.IsInRing()),
    ]

# -------------------------
# SMILES → Graph
# -------------------------
def smiles_to_graph(smiles, label):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Node features
    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    # Edges
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    # 🔥 FIX: handle molecules with no bonds
    if len(edges) == 0:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # -------------------------
    # Fingerprint (CONSISTENT)
    # -------------------------
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp = np.array(fp)
    fp = torch.tensor(fp, dtype=torch.float)

    # Label
    y = torch.tensor([label], dtype=torch.float)

    # Create graph
    data = Data(x=x, edge_index=edge_index, y=y)

    # Attach fingerprint
    data.fp = fp

    return data

# -------------------------
# Load dataset
# -------------------------
def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    # Clean labels
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]

    graphs = []

    for _, row in df.iterrows():
        graph = smiles_to_graph(row["smiles"], float(row["label"]))
        if graph is not None:
            graphs.append(graph)

    return graphs

# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    graphs = load_dataset("data/combined.csv")
    print("Graphs:", len(graphs))
    print(graphs[0])