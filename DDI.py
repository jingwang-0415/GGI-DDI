import torch
import torch.nn.functional as F
from torch import nn
from  interaction import Interactions
from torch_geometric.nn import GINEConv
from Drug_Conv import ASAP_Pooling
class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64):
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.line_x = nn.Linear(in_dim, hidden_dim)
        self.line_edge = nn.Linear(edge_in_dim,hidden_dim)
        self.conv1 = GINEConv(mlp)

    def forward(self, data):
        x = self.line_x(data.x)
        edge_attr = self.line_edge(data.edge_attr)
        edge_attr = (x[data.edge_index[0]] + x[data.edge_index[1]] + edge_attr) / 3
        x = self.conv1(x,data.edge_index,edge_attr)
        edge_attr = (x[data.edge_index[0]] + x[data.edge_index[1]] + edge_attr) / 3

        return x,edge_attr


class DD_Pre(torch.nn.Module):
    def __init__(self, in_channels, ratio1, ratio2):
        super(DD_Pre, self).__init__()

        hidden_dim = 64


        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.tr_inter = nn.Sequential(
            nn.Embedding(1318,in_channels),

            nn.PReLU(),
            nn.Linear(in_channels,hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim,hidden_dim),
        )
        self.loss_fn = nn.BCELoss()
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio2
        self.dropout = 0.2
        self.conv1 = DrugEncoder(in_channels,6,hidden_dim)


        #
        self.conv2 = GINEConv(mlp2,edge_dim=hidden_dim)

        self.pool1 = ASAP_Pooling(hidden_dim, self.ratio1,hidden_dim,edge_hidden=hidden_dim)

        self.pool2 = ASAP_Pooling(hidden_dim, self.ratio3,hidden_dim,edge_hidden=hidden_dim)
        self.Interactions = Interactions(hidden_dim)#.to(device)

    def forward(self,heads,tails,rels,site):
        rels = self.tr_inter(rels)
        x_1,edge_attr_1 = self.conv1(heads)
        x_2,edge_attr_2 = self.conv1(tails)

        X_1_1, edge_1_1, batch_1_1,perm_1_1,fitness_1_1,index_S_1_1,value_S_1_1,edge_attr_1_1 = self.pool1(x_1, heads.edge_index, heads.batch,edge_attr_1)
        X_2_1, edge_2_1, batch_2_1,perm_2_1,fitness_2_1,index_S_2_1,value_S_2_1,edge_attr_2_1= self.pool1(x_2, tails.edge_index,tails.batch,edge_attr_2)

        x_1_2 = F.relu(F.dropout(self.conv2(X_1_1, edge_1_1,edge_attr_1_1),p=self.dropout,training=site))
        x_2_2 = F.relu(F.dropout(self.conv2(X_2_1, edge_2_1,edge_attr_2_1),p=self.dropout,training=site))
        edge_attr_1_1 = (x_1_2[edge_1_1[0]] + x_1_2[edge_1_1[1]] + edge_attr_1_1)/3
        edge_attr_2_1 = (x_2_2[edge_2_1[0]] + x_2_2[edge_2_1[1]] + edge_attr_2_1)/3

        X_1_2, edge_1_2, batch_1_2, perm_1_2, fitness_1_2, index_S_1_2, value_S_1_2, edge_weight_1_2 = self.pool2(x_1_2, edge_1_1,batch_1_1,edge_attr_1_1)
        X_2_2, edge_2_2, batch_2_2, perm_2_2, fitness_2_2, index_S_2_2, value_S_2_2, edge_weight_2_2 = self.pool2(x_2_2, edge_2_1, batch_2_1,edge_attr_2_1)

        logit,pos_1,pos_2 = self.Interactions(X_1_2,batch_1_2, X_2_2,batch_2_2,rels)

        return logit

