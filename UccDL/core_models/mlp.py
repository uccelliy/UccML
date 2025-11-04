import torch 
from torch import nn

class fc_net(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(fc_net, self).__init__()
        layers = []
        self.hidden_num = len(hidden_sizes)
        for i in range(self.hidden_num):
            if i == 0:
                layer = nn.Linear(input_size, hidden_sizes[i])
            else:
                layer = nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            layers.append(layer)
            if i != self.hidden_num - 1:
                layers.append(nn.ReLU())
        self.network=nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)