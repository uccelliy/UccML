import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wcombine=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)
        
    def forward(self,q,k,v,mask=None):
        batchm,time,dimension=q.shape
        n_d=self.d_model//self.num_heads
        q,k,v=self.wq(q),self.wk(k),self.wv(v)
        q=q.view(batchm,time,self.num_heads,n_d).permute(0,2,1,3)
        k=k.view(batchm,time,self.num_heads,n_d).permute(0,2,1,3)
        v=v.view(batchm,time,self.num_heads,n_d).permute(0,2,1,3)
        score=q@k.transpose(2,3)/math.sqrt(n_d)
        if mask is not None:
            score=score.masked_fill(mask==0,-100000)
        score=self.softmax(score)@v
        score=score.permute(0,2,1,3).contiguous().view(batchm,time,dimension)
        out=self.wcombine(score)
        return out
    