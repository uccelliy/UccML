import torch
from torch import nn
import multiheadAttention as MultiHeadAttention
import layernorm as LayerNorm
import embedding as Embedding

class PositioanwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositioanwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.dropout(torch.relu(self.fc1(x)))
        x=self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PositioanwiseFeedForward(d_model,ffn_hidden,dropout)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        _x=x
        x=self.self_attn(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x
    
class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,dropout=0.1,device):
        super(Encoder,self).__init__()
        self.embedding=Embedding.TeanserEmbedding(enc_voc_size,d_model,max_len,dropout,device)
        self.layers=nn.ModuleList([EncoderLayer(d_model,ffn_hidden,n_head,device) for _ in range(n_layers)])
        
    def forward(self,x,mask=None):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,mask)
        return x
        
        
        