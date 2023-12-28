import torch
from torch import nn
from typing import Tuple

from tsl.nn.blocks.encoders import TemporalConvNet, Transformer
from tsl.nn.blocks.encoders.recurrent import RNN, ConditionalBlock
from tsl.nn.blocks.decoders import GCNDecoder

from utils import init_weights_xavier, round_to_nearest_divisible

activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU,
    'selu': nn.SELU,
    'leaky_relu': nn.LeakyReLU,
}

encoders = {
    'rnn': RNN,
    'tcn': TemporalConvNet, 
    'transformer': Transformer,
}

class UniModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = TemporalConvNet()
        self.decoder = GCNDecoder()

        self.encoder.apply(init_weights_xavier)
        self.decoder.apply(init_weights_xavier)

    def forward(self, x, edges, weights):
        x = self.encoder(x) 
        x = self.decoder(x, edges, weights)
        return x

class BiModel(nn.module):
    def __init__(self):
        super().__init__()

        self.model_f = UniModel()
        self.model_b = UniModel()
        self.cond_block = ConditionalBlock()
        self.decoder_mlp = nn.Sequential().apply(init_weights_xavier)

        print(self.model_f)
        print(self.decoder_mlp)
   
    def bi_forward(self, input_tensor, edges, weights, co_tensor):
        f_repr = self.model_f(input_tensor, edges, weights)
        b_repr = self.model_b(torch.flip(input_tensor, dims=[1]), edges, weights)

        h = torch.cat([f_repr, b_repr], dim=-1)

        h = self.cond_block(h, u = co_tensor)
        return self.decoder_mlp(h)
    
    def forward(self, x_t, mask, edges, weights, t, u, x_co):
        t_tensor = torch.ones_like(x_t) * t

        input_tensor = torch.cat([x_t, x_co, mask, t, u], dim=-1)
        cond_tensor = torch.cat([t_tensor, u], dim=-1)

        input_tensor = torch.stack(input_tensor).permute(1, 2, 3, 0)
        co_tensor = torch.stack(cond_tensor).permute(1, 2, 3, 0)
     
        res = self.bi_forward(input_tensor, edges, weights, co_tensor).squeeze(dim=-1)
        return res
