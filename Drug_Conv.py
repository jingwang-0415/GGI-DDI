import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import nn
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import GCNConv, GraphConv, LEConv
# from LE_Conv import LEConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, degree
from torch_geometric.nn.pool.topk_pool import topk
from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm
import math
from InterGAT import InterGAT
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree

def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""

    kN = perm.size(0)
    perm2 = perm.view(-1, 1)

    # mask contains bool mask of edges which originate from perm (selected) nodes
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)

    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_attr=value_E,
                                                fill_value=fill_value, num_nodes=kN)

    return index_E, value_E, index_S, value_S


class ASAP_Pooling(torch.nn.Module):

    def __init__(self, in_channels, ratio, edge_dim, edge_hidden, disc_hidden=0, dropout_att=0, negative_slope=0.2,
                 heads=1):
        super(ASAP_Pooling, self).__init__()

        self.ln0 = nn.LayerNorm(in_channels)
        self.ln1 = nn.LayerNorm(in_channels)

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.loss_fn = nn.BCELoss()
        self.heads = heads

        self.lin_x = Linear(in_channels, in_channels)
        self.lin_v = Linear(in_channels, in_channels)
        self.readout_linear = nn.Linear(in_channels, in_channels)
        self.out_linear = nn.Linear(in_channels, in_channels)

        self.gnn_score = LEConv(self.in_channels, 1)
        self.gnn_intra_cluster = InterGAT(edge_dim, in_channels, in_channels, edge_hidden=edge_hidden,
                                          heads=2)
        self.reset_parameters()

    def reset_parameters(self):

        self.gnn_score.reset_parameters()

        self.gnn_intra_cluster.reset_parameters()
        self.lin_x.reset_parameters()
        self.readout_linear.reset_parameters()
        self.lin_v.reset_parameters()
        self.out_linear.reset_parameters()


    def forward(self, x, edge_index, batch, edge_weight=None, x_readout=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)

        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_attr=edge_weight,
                                                           num_nodes=num_nodes.sum())

        N = x.size(0)  # total num of nodes in batch
        out, score = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_attr=edge_weight)
        if x_readout == None:
            x_readout = global_add_pool(x, batch)
        x_readout = x_readout.repeat_interleave(degree(batch, dtype=batch.dtype), dim=0)


        x_j = out[edge_index[1]]

        v_j = x_j * score.view(-1, 1)
        # # ---Aggregation---
        # # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)

        score_local = torch.sigmoid(self.gnn_score(x=out, edge_index=edge_index).view(-1))
        x_readout = self.readout_linear(x_readout)
        out_liner = self.out_linear(out)
        score_glo = torch.sigmoid(((x_readout * out_liner).sum(dim=-1))).view(-1)

        # ---Cluster Selection
        # Nx1
        fitness = score_local+score_glo
        perm = topk(x=fitness, ratio=self.ratio, batch=batch)
        # x = out[perm]
        X = out[perm] * fitness[perm].view(-1, 1)
        # ---Maintaining Graph Connectivity
        batch = batch[perm]

        edge_weight = None
        edge_index, edge_weight, index_S, value_S = graph_connectivity(
            device=x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=score,
            ratio=self.ratio,
            batch=batch,
            N=N)
        x = self.lin_x(X)
        edge_weight = ((x[edge_index[0]] + x[edge_index[1]]) / 2) * edge_weight.view(-1, 1)


        return X, edge_index, batch, perm, fitness, index_S, value_S, edge_weight

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)
