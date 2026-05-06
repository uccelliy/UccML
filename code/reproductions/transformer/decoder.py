import torch
from torch import nn
from torch.nn import functional as F
import multiheadAttention as MultiAttention
import layernorm as LayerNorm
import embedding as Embedding
import encoder as Encoder
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.attention1=MultiAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.cross_attention=MultiAttention(d_model,n_head)
        self.dropout2=nn.Dropout(dropout)
        self.ffn=Encoder.PositioanwiseFeedForward(d_model,ffn_hidden,dropout)
        self.norm3=LayerNorm(d_model)
        self.dropout3=nn.Dropout(dropout)
        
    def forward(self,dec,enc,t_mask,s_mask):
        _x=dec
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.cross_attention(x,enc,enc,s_mask)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm3(x+_x) 
        return x
    
class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,device,dropout=0.1):
        super(Decoder,self).__init__()
        self.embedding=Embedding.TransformerEmbedding(dec_voc_size,d_model,max_len,dropout,device)
        self.layers=nn.ModuleList([DecoderLayer(d_model,ffn_hidden,n_head,dropout) for _ in range(n_layers)])
        self.fc=nn.Linear(d_model,dec_voc_size)
        
    def forward(self,dec,enc,s_mask,t_mask):
        dec=self.embedding(enc)
        for layer in self.layers:
            dec=layer(dec,enc,s_mask,t_mask)
        dec=self.fc(dec)
        return dec
        