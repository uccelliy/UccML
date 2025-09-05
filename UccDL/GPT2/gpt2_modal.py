import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
print(torch.__version__)
print(torch.cuda.is_available())

tranform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,download=True,transform=tranform)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)