import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data
from model.model import ToxicityGNN
import numpy as np
import os

# -------------------------
# Atom features
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
# Functional groups
# -------------------------
def detect_groups(mol):
    groups = []

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
        groups.append("nitro")

    if mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")):
        groups.append("phenol")

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OH]")):
        groups.append("alcohol")

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3]")):
        groups.append("amine")

    if mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1")):
        groups.append("aromatic")

    return groups

# -------------------------
# Mechanism mapping 🔬
# -------------------------
def group_to_mechanism(group):
    mapping = {
        "nitro": "metabolic activation leading to reactive intermediates",
        "phenol": "protein denaturation and oxidative stress",
        "alcohol": "dose-dependent CNS and metabolic effects",
        "amine": "interaction with biological receptors and pH balance",
        "aromatic": "increased lipophilicity and bioaccumulation potential"
    }
    return mapping.get(group, "")

# -------------------------
# Structured reasoning 🧠
# -------------------------
def build_reasoning(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "name": "Invalid",
            "groups": [],
            "mechanisms": []
        }

    groups = detect_groups(mol)
    mechanisms = [group_to_mechanism(g) for g in groups]

    return {
        "groups": groups,
        "mechanisms": mechanisms
    }

# -------------------------
# SMILES → Graph
# -------------------------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp = torch.tensor(np.array(fp), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.fp = fp

    return data

# -------------------------
# Load model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ToxicityGNN(num_node_features=9).to(device)

# Get the directory of this file and build the path to the model
model_path = os.path.join(os.path.dirname(__file__), "toxicity_gnn_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded ✅")

# -------------------------
# Predict & SHAP Insights
# -------------------------
def predict_single(smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None, None, []

    graph = graph.to(device)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(graph).view(-1)
        prob = torch.sigmoid(out).item()

    label = "TOXIC" if prob > 0.5 else "NON-TOXIC"

    # SHAP Explainability for Fingerprint Features
    shap_insights = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        bit_info = {}
        if mol is not None:
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)

        import shap
        class SHAPModelWrapper(torch.nn.Module):
            def __init__(self, base_model, graph_x, edge_index, batch):
                super().__init__()
                self.base_model = base_model
                self.graph_x = graph_x
                self.edge_index = edge_index
                self.batch = batch
            def forward(self, fp):
                from torch_geometric.data import Data
                data = Data(x=self.graph_x, edge_index=self.edge_index, batch=self.batch)
                data.fp = fp
                return self.base_model(data)

        wrapper = SHAPModelWrapper(model, graph.x, graph.edge_index, graph.batch)
        wrapper.train()
        
        bg = torch.zeros((1, 2048), dtype=torch.float).to(device)
        test_fp = graph.fp.view(1, 2048).to(device)
        test_fp.requires_grad_()
        
        explainer = shap.GradientExplainer(wrapper, bg)
        shap_values = explainer.shap_values(test_fp)
        
        vals = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
        
        import numpy as np
        from rdkit.Chem import PathToSubmol, FindAtomEnvironmentOfRadiusN, MolToSmiles
        
        top_indices = np.argsort(vals)[-3:][::-1]
        for idx in top_indices:
            if vals[idx] > 0.01:
                # Map SHAP bit back to the chemical Substructure SMILES!
                if mol is not None and idx in bit_info:
                    atom_idx, radius = bit_info[idx][0]
                    if radius == 0:
                        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                        shap_insights.append(f"Atom '{symbol}' (SHAP importance score: {vals[idx]:.4f}) [mapped from Bit {idx}]")
                    else:
                        env = FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                        amap = {}
                        submol = PathToSubmol(mol, env, atomMap=amap)
                        frag_smiles = MolToSmiles(submol)
                        shap_insights.append(f"Substructure '{frag_smiles}' (SHAP importance score: {vals[idx]:.4f}) [mapped from Bit {idx}]")
                else:
                    shap_insights.append(f"Molecular Fingerprint Bit {idx} (SHAP importance score: {vals[idx]:.4f})")
    except Exception as e:
        shap_insights = [f"SHAP Error: {str(e)}"]

    return label, prob, shap_insights

# -------------------------
# MAIN FUNCTION 🔥
# -------------------------
def predict_smiles(smiles1, smiles2):

    res1, conf1, shap1 = predict_single(smiles1)
    res2, conf2, shap2 = predict_single(smiles2)

    if res1 is None or res2 is None:
        return {"error": "Invalid SMILES"}

    avg_conf = (conf1 + conf2) / 2

    # Risk logic
    if res1 == "TOXIC" and res2 == "TOXIC":
        interaction = "HIGH RISK ⚠️"
    elif res1 != res2:
        interaction = "MEDIUM RISK ⚠️"
    else:
        interaction = "LOW RISK ✅"

    # 🧠 Structured reasoning
    A = build_reasoning(smiles1)
    A["shap_insights"] = shap1
    B = build_reasoning(smiles2)
    B["shap_insights"] = shap2

    reasoning_struct = {
        "drugA": A,
        "drugB": B,
        "interaction_reason": "Potential additive or synergistic toxic effects due to overlapping mechanisms."
    }

    return {
        "drugA": {"smiles": smiles1, "prediction": res1, "confidence": round(conf1, 4)},
        "drugB": {"smiles": smiles2, "prediction": res2, "confidence": round(conf2, 4)},
        "interaction": interaction,
        "overall_confidence": round(avg_conf, 4),
        "structured_reasoning": reasoning_struct
    }