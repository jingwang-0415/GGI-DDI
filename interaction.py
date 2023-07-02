import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool

class Interactions(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bias = nn.Parameter(torch.zeros(in_channels))
        # self.tr_inter = nn.Sequential(
        #     nn.PReLU(),
        #     nn.Linear(self.in_channels, self.in_channels),
        #     nn.PReLU(),
        #     nn.Linear(self.in_channels, self.in_channels)
        # )
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_channels*2, self.in_channels*2),
            nn.PReLU(),
            nn.Linear(self.in_channels*2, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels , self.in_channels),
        )
        # self.weight_1 = nn.Parameter(torch.zeros(in_channels, in_channels))
        # self.weight_2 = nn.Parameter(torch.zeros(in_channels, in_channels))
        self.weight_1 = nn.Sequential(
            nn.Linear(self.in_channels*2, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, 1),
            nn.Sigmoid()
        )
        self.weight_2 = nn.Sequential(
            nn.Linear(self.in_channels*2, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, 1),
            nn.Sigmoid()

        )
        self.a = nn.Parameter(torch.zeros(in_channels))
        # glorot(self.weight_1)
        # glorot(self.weight_2)
        glorot(self.a.view(1, -1))
    def forward(self, x_1, batch_1, x_2, batch_2, inter):
        x_1_readout = global_add_pool(x_1,batch_1)
        x_2_readout = global_add_pool(x_2,batch_2)
        d_1 = degree(batch_1, dtype=batch_1.dtype)
        d_2 = degree(batch_2, dtype=batch_2.dtype)
        x_1_readout = x_1_readout.repeat_interleave(d_2,dim=0)
        x_2_readout = x_2_readout.repeat_interleave(d_1,dim=0)


        x_1_score = torch.cat([x_1,x_2_readout],dim=1)
        x_2_score = torch.cat([x_2,x_1_readout],dim=1)
        x_1 = x_1 * self.weight_1(x_1_score)
        x_2 = x_2 * self.weight_2(x_2_score)

        s_1 = torch.cumsum(d_1, dim=0)
        s_2 = torch.cumsum(d_2, dim=0)
        ind_1 = torch.cat([torch.arange(i,device = d_1.device).repeat_interleave(j) + (s_1[e - 1] if e else 0) for e, (i, j) in enumerate(zip(d_1, d_2))])
        ind_2 = torch.cat([torch.arange(j,device = d_1.device).repeat(i) + (s_2[e - 1] if e else 0) for e, (i, j) in enumerate(zip(d_1, d_2))])
        x_1 = x_1[ind_1]
        x_2 = x_2[ind_2]
        size_1_2=torch.mul(d_1,d_2)
        inputs = torch.cat((x_1, x_2), 1)
        ans_SSI = (self.a * self.mlp(inputs))
        ans_SSI = (ans_SSI * inter.repeat_interleave(size_1_2, dim=0)).sum(-1)

        batch_ans = torch.arange(inter.shape[0],device = inputs.device).repeat_interleave(size_1_2, dim=0)
        ans = scatter(ans_SSI,batch_ans,reduce='sum',dim=0)
        
        logit = ans

        # ans = torch.cat([ans,x_1_out,x_2_out],dim=1)
        return logit,x_1,x_2










