import torch
from torch import nn

class conv_net(nn.Module):
    def __init__(self, input_channels,input_size, conv_layers_config):
        super(conv_net, self).__init__()
        layers = []
        in_channels = input_channels
        self.in_size = input_size
        for config in conv_layers_config:
            layers.append(nn.Conv2d(in_channels, config['out_channels'], kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding']))
            layers.append(nn.ReLU())
            if 'pool_kernel_size' in config:
                layers.append(nn.MaxPool2d(kernel_size=config['pool_kernel_size'], stride=config['pool_stride']))
            in_channels = config['out_channels'] 
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x= self.network(x)
        x = self.flatten(x)
        return x
    
    def get_conv_out(self):
        with torch.no_grad():
            input_tensor = torch.zeros(1, *((self.network[0].in_channels,) + self.in_size))
            output_tensor = self.network(input_tensor)
            return output_tensor.numel() //output_tensor.size(0)
    
class resn_net(nn.Module):
    def __init__(self, input_channels, input_size, resnet_config):
        super(resn_net, self).__init__()
        layers = []
        in_channels = input_channels
        self.in_size = input_size
        for config in resnet_config:
            stride = config['stride']
            out_channels = config['out_channels']
            num_blocks = config['num_blocks']
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )
            layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))   
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.network(x)
        x = self.flatten(x)
        return x   
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
