import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .layers.encoder_layers import *

class SumCapEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_inp=1024, n_layers=4, n_head=1, d_k=64, d_v=64,
            d_model=256, d_inner=512, dropout=0.1, use_drop_out=False, use_layer_norm=False):

        super().__init__()
        self.n_layers = n_layers
        self.use_drop_out = use_drop_out
        self.use_layer_norm = use_layer_norm
        
        self.proj = None
        self.proj = nn.Linear(d_inp, d_model) 
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            SumCapEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.linear_1 = nn.Linear(in_features=d_inp, out_features=d_inp)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, src_seq):
        # -- Forward
        enc_output = self.proj(src_seq)
        enc_output = self.layer_norm(enc_output)
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, _, s = enc_layer(enc_output)
        
        # 2 layer NN for regressor
        s = self.linear_1(s)
        s = self.relu(s)
        
        if self.use_drop_out == True:
            s = self.drop(s)
        if self.use_layer_norm == True:
            s = self.norm_linear(s)

        s = self.linear_2(s)
        s = self.sigmoid(s)
        s = s.squeeze(-1)
            
        return enc_output, s

if __name__ == '__main__':
    model = SumCapEncoder()
    inp = torch.rand(1,300,1024)
    enc_output = model(inp)
    print(enc_output.shape)
