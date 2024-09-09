from torch import nn
import torch.nn.functional as F
import math
from torch import Tensor 

class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_modal):
        super(TokenEmbedding,self).__init__(vocab_size,d_modal,padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self,d_modal,max_len,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_modal,device=device)
        self.encoding.requires_grad=False
        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        _2i=torch.arange(0,d_modal,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_modal)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_modal)))

    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]

class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_modal,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_modal)
        self.pos_emb=PositionalEmbedding(d_modal,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)
        def forward(self.x):
            tok_emb=self.tok_emb(x)
            pos_emb=self.pos_emb(x)
            return self.drop_out(tok_emb+pos_emb)


