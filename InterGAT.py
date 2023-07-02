import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.utils import  softmax,add_self_loops,remove_self_loops
from torch_geometric.nn.inits import glorot
from torch.nn import Parameter
import torch
from torch import  Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import (
    Optional,
)
class InterGAT(MessagePassing):
    def __init__(self, edge_dim, n_feats,hidden_dim,edge_hidden=16,heads=2):
        super().__init__()
        self.heads = heads
        self.hidden = hidden_dim
        self.edge_hidden = edge_hidden
        self.lin_node = nn.Linear(n_feats,heads* hidden_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim,heads* edge_hidden, bias=False)
        # self.lin_fuse = nn.Linear((edge_hidden+hidden_dim)*heads,hidden_dim*heads)
        # self.lin_fuse = nn.Linear((hidden_dim*2)*heads,hidden_dim*heads)

        self.lin_fuse = nn.Linear((edge_hidden+hidden_dim*2)*heads,hidden_dim*heads)
        self.out_node = nn.Linear(hidden_dim,1,bias=True)
        # self.attention_self = attention_self(hidden_dim=hidden_dim,heads=heads)
        self.att_fuse_node = Parameter(torch.Tensor(1,heads,hidden_dim*2+edge_hidden))
        # self.att_fuse_edge = Parameter(torch.Tensor(1,heads,hidden_dim*3))
        # self.edge_fuse = nn.Linear(hidden_dim*4,hidden_dim)
        # self.node_fuse_2 = nn.Linear(hidden_dim*2,hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        # self.out_node.reset_parameters()
        # self.edge_fuse.reset_parameters()
        # self.node_fuse_2.reset_parameters()
        glorot(self.att_fuse_node)


    def forward(self, x,edge_index,edge_attr):
        num_nodes = x.size(0)
        # edge_index,edge_attr = remove_self_loops(edge_index,edge_attr=edge_attr)
        # edge_index, edge_attr = add_self_loops(edge_index,edge_attr=edge_attr,num_nodes=num_nodes)

        x = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)
        # new_x , new_edge_attr = self.attention_self(x=x,edge_index=edge_index,edge_attr=edge_attr)
        x = (x, x)

        x_src,x_dst = x[0],x[1]
        out,score = self.propagate(edge_index,x=(x_src,x_dst),edge_attr=edge_attr)

        # out = (x[0]+ self.lin_fuse(out)).view(-1,self.heads,self.hidden)

        out = self.lin_fuse(torch.cat([x[0],out],dim=1)).view(-1,self.heads,self.hidden)
        out = torch.mean(out,dim=1)
        # _,score = add_self_loops(edge_index,edge_attr=score,num_nodes=num_nodes,fill_value=0.)
        # out = torch.cat([out,x[1].view(-1,self.heads,self.hidden).mean(dim=1)],dim=1)
        # out = self.out_node(out)
        return out,score.unsqueeze(1)

    def message(self, x_j: Tensor,x_i,edge_attr,index) -> Tensor:
        fuse_message = torch.cat([x_j,x_i,edge_attr],dim=1)
        fuse_message = F.leaky_relu(fuse_message.view(-1,self.heads,self.hidden*2+self.edge_hidden),negative_slope=0.2)
        node_alpha = (fuse_message * self.att_fuse_node).sum(-1)
        node_alpha = softmax(node_alpha, index)
        # x_new = torch.cat([x_j],dim=1).view(-1,self.heads,self.hidden) * node_alpha.unsqueeze(-1)

        x_new = torch.cat([x_j,edge_attr],dim=1).view(-1,self.heads,self.hidden+self.edge_hidden) * node_alpha.unsqueeze(-1)
        x_new = x_new.view(-1,self.heads*(self.hidden+self.edge_hidden))
        # x_new = x_new.view(-1,self.heads*(self.hidden))

        return x_new,node_alpha
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        x_new,weight = inputs
        x_new = super().aggregate(x_new,index)
        return x_new,weight.mean(dim=1)
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

if __name__ == '__main__':
    a = InterGAT(1,1,2,2,2)
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_attr = torch.tensor([[1],[2],[3],[4]],dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(),edge_attr=edge_attr)
    a(data.x,data.edge_index,data.edge_attr)