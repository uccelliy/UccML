import torch
from torch import nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,unbiased=False,keepdim=True)
        return self.gamma*(x-mean)/torch.sqrt(std**2+self.eps)+self.beta