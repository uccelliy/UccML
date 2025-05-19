from IPython import display
import torch
from d2l import torch as d2l


batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs=784
num_outputs=10
W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)



def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition

#X=torch.normal(0,1,(2,5))
#X_prob=softmax(X)
#print(X_prob, X_prob.sum(1,keepdim=True))

def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    if len(y_hat.shape) >1 and y_hat.shape[1] >1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())