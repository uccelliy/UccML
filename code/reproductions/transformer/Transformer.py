import torch
from torch import nn
import torch.nn.functional as F
import math
import encoder as Encoder
import decoder as Decoder

class Transformer(nn.Module):
    def __init__(self,src_pad_ix,trg_pad_ix,enc_voc_size,dec_voc_size,max_len,d_modal,n_heads,ffn_hidden,n_layers,drop_prob,device):
        super(Transformer,self).__init__()
        self.encoder=Encoder(self,enc_voc_size,max_len,d_modal,ffn_hidden,n_heads,n_layers,device,drop_prob)
        self.decoder=Decoder(self,dec_voc_size,max_len,d_modal,ffn_hidden,n_heads,n_layers,device,drop_prob)
        self.src_pad_ix=src_pad_ix
        self.trg_pad_ix=trg_pad_ix
        self.device=device

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1) 
        mask=q&k
        return mask
    
    def mask_casual_mask(self,p,k):
        len_q,len_k=q.size(1),k.size(1)
        mask=torch.trill(torch.ones(len_q,len_k).type(torch.BoolTensor).to(self.device))
        return mask
        
    def forward(self, src, trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_ix,self.src_pad_ix)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_ix,self.trg_pad_ix)*self.mask_casual_mask(trg,trg)
        enc_output=self.encoder(src,src_mask)
        dec_output=self.decoder(trg,enc_output,trg_mask,src_mask)
        return dec_output
