import torch
import sys
sys.path.append('c:/Users/radhe/Desktop/toxlens-ai-main/backend')
from model.model import ToxicityGNN
from torch_geometric.data import Data
import shap
import numpy as np

# Load model
device = torch.device('cpu')
model = ToxicityGNN(num_node_features=9).to(device)

# Create dummy graph wrapper for SHAP
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, edge_index, batch):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.batch = batch
        
    def forward(self, x, fp):
        data = Data(x=x, edge_index=self.edge_index, batch=self.batch)
        data.fp = fp
        return self.model(data)

x = torch.randn(5, 9)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
batch = torch.zeros(5, dtype=torch.long)
fp = torch.randn(1, 2048)

wrapper = ModelWrapper(model, edge_index, batch)

try:
    # SHAP needs background distribution
    bg_x = torch.randn(10, 5, 9)
    bg_fp = torch.randn(10, 1, 2048)
    
    # We might need to wrap it differently since x is (num_nodes, features)
    print("Testing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(wrapper, [bg_x[0], bg_fp[0]])
    shap_values = explainer.expected_value
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
