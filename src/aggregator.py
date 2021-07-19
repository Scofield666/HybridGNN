import torch
from torch import nn
import math
from torch.nn import functional as F
from torch.autograd import Variable


class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanAggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.param_reset(output_dim)


    def param_reset(self, embed_dim):
        self.fc_x.weight.data.normal_(std=1.0 / math.sqrt(embed_dim))
        self.fc_neib.weight.data.normal_(std=1.0 / math.sqrt(embed_dim))
    '''
        x: batch * embedding
        neibs: (batch * N) * embedding
        => 
        agg_neib: batch * N * embedding
        
        @:return batch * embedding
    '''
    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.mean(agg_neib, dim=1)
        out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        out = self.activation(out)
        return out

