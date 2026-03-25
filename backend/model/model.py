import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class ToxicityGNN(torch.nn.Module):

    def __init__(self, num_node_features, hidden_dim=64, heads=4, dropout=0.3):
        super(ToxicityGNN, self).__init__()

        # GNN layers
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, 128, heads=heads)

        # ✅ FIXED fingerprint layer (2048 → 64)
        self.fp_layer = torch.nn.Linear(2048, hidden_dim)

        # combined layers
        self.fc1 = torch.nn.Linear(128 * heads + hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

        # dropout
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # number of graphs in batch
        num_graphs = batch.max().item() + 1

        # ✅ FIXED reshape (important)
        fp = data.fp.view(num_graphs, 2048)

        # GNN
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        # fingerprint
        fp = self.fp_layer(fp)

        # combine
        x = torch.cat([x, fp], dim=1)

        # dense
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x