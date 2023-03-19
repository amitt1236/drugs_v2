import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, arch):
        super(GNNEncoder, self).__init__()
        self.arch = arch
        # creating a list of gnn convs layers
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        # creating a list of Layer Normalization
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for _ in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim//2), nn.Dropout(0.25),
            nn.Linear(output_dim//2, output_dim))


        self.dropout = 0.25
        self.num_layers = 3
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers
        if self.arch == 'GAT':
            return pyg_nn.GATConv(input_dim, hidden_dim, heads=1, dropout=0.2, edge_dim=3)
        else:
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        
    def forward(self, data):
        x, edge_index, edge_atrr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Applying conv
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_atrr)  
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        
        # post message-passing MLP
        x = self.post_mp(x)

        # Pooling node features 
        x = pyg_nn.global_mean_pool(x,batch)

        return x
    
    def loss(self, graph_features, text_features, labels):
        # normalized features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        total_loss = (F.cross_entropy(logits_per_graph, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss
