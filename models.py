"""File containing the model of the Graph Attentive Vectors link prediction framework."""

import torch
from torch import nn
from torch.nn import (Linear, ModuleList, Conv1d, MaxPool1d)
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class GAV(torch.nn.Module):
    """The Graph Attentive Vectors link prediction framework"""
    def __init__(self, num_layers=1):
        super().__init__()

        # Message-passing
        self.convs = ModuleList()
        self.convs.append(GAVLayer(in_dim=5, h_dim=32))
        for _ in range(0, num_layers - 1):
            self.convs.append(GAVLayer(in_dim=5, h_dim=32))

        # Sort Pooling
        self.sp = False
        conv1d_channels = [16, 32]
        total_latent_dim = 5 * (num_layers) # TODO 
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((10 - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]     

        # Readout module
        self.lin1 = Linear(10, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        _, ids = x[:, :3], x[:, 3:]
        xs = [x]

        # Message-passing module
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        # Readout
        x_fp = torch.cat((xs[-1], ids), dim=-1)
        batch_ranges = batch.unique(return_counts=True)[1].tolist()
        x_split = torch.split(x_fp, batch_ranges)

        src_flow = []
        dst_flow = []
        for x_batch in x_split:
            src_flow.append(x_batch[x_batch[:, -2] == 1][:, :5].mean(dim=0))
            dst_flow.append(x_batch[x_batch[:, -1] == 1][:, :5].mean(dim=0))

        flow = torch.cat((torch.stack(src_flow), torch.stack(dst_flow)), dim=-1)

        # MLP
        x = F.relu(self.lin1(flow))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    

class GAVLayer(nn.Module):
    """The GAV message-passing layer"""
    def __init__(self, in_dim=3, h_dim=32, out_dim=None):
        super().__init__()
        self.attn_in_proj = Linear(in_dim, h_dim)
        self.attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=4)
        self.attn_dropout = nn.Dropout(0.0)
        self.attn_norm = nn.LayerNorm(h_dim)
        self.attn_out_mlp = MLP(h_dim, h_dim*2, 1, 2, act='lrelu')

        if out_dim is not None:
            self.increase_dim = True
            self.out_proj = Linear(in_dim, out_dim)
        else:
            self.increase_dim = False

    def gen_mask(self, edge_index):
        A = to_dense_adj(edge_index).bool()[0]
        A.fill_diagonal_(True)  # add self-loop
        attn_mask = ~A
        return attn_mask

    def forward(self, x, edge_index):
        x_proj = F.leaky_relu(self.attn_in_proj(x))
        x_attn = self.attn(x_proj, x_proj, x_proj, attn_mask=self.gen_mask(edge_index))[0]
        x_skip = x_proj + self.attn_dropout(x_attn)
        x_norm = self.attn_norm(x_skip)
        x_out = self.attn_out_mlp(x_norm)
        x_pred = x_out.tanh()

        x = x_pred * x  # multiply scalar

        if self.increase_dim:
            x = F.leaky_relu(self.out_proj(x))
        return x


class MLP(torch.nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='lrelu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        if act == 'relu':
            self.act = F.relu
        elif act == 'lrelu':
            self.act = F.leaky_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
