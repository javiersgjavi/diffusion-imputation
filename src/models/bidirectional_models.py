import torch
from torch import nn
from typing import Tuple

from tsl.nn.blocks.encoders import TemporalConvNet, Transformer
from tsl.nn.blocks.encoders.recurrent import RNN
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from tsl.nn.blocks.decoders import GCNDecoder

from src.models.utils import init_weights_xavier, round_to_nearest_divisible

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

        self.cond1 = ConditionalBlock(
            input_size=3,
            exog_size=128,
            output_size=3,
            activation='relu'
        )
        
        self.encoder = TemporalConvNet(
            input_channels=3,
            hidden_channels=30,
            kernel_size=7,
            dilation=5,
            stride=1,
            n_layers=3,
            dropout=0.1,
        )

        self.cond2 = ConditionalBlock(
            input_size=30,
            exog_size=128,
            output_size=30,
            activation='relu'
        )

        self.decoder = GCNDecoder(
            input_size=30,
            hidden_size=20,
            output_size=10,
            horizon=24,
            n_layers=2
        )

        self.encoder.apply(init_weights_xavier)
        self.decoder.apply(init_weights_xavier)
        self.cond1.apply(init_weights_xavier)
        self.cond2.apply(init_weights_xavier)

    def forward(self, x, edges, weights, cond_tensor):
        x = self.cond1(x, cond_tensor)
        x = self.encoder(x) 
        x = self.cond2(x, cond_tensor)
        x = self.decoder(x, edges, weights)
        return x

class BiModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_f = UniModel()
        self.model_b = UniModel()
        self.cond = ConditionalBlock(
            input_size=20,
            exog_size=128,
            output_size=20,
            activation='tanh'
        ).apply(init_weights_xavier)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Tanh()
            
            #nn.Tanh()
        ).apply(init_weights_xavier)

        #print(self.model_f)
        #print(self.decoder_mlp)
   
    def bi_forward(self, input_tensor, edges, weights, cond_tensor):
        f_repr = self.model_f(input_tensor, edges, weights, cond_tensor)
        b_repr = self.model_b(torch.flip(input_tensor, dims=[1]), edges, weights, cond_tensor)

        h = torch.cat([f_repr, b_repr], dim=-1)

        h = self.cond(h, cond_tensor)

        pred = self.decoder_mlp(h)
        return pred
    
    def forward(self, x_t, x_co, mask, edges, weights, t_emb, u):

        u = u.unsqueeze(2).repeat(1, 1, mask.shape[2], 1)

        input_tensor = torch.cat([x_t, x_co, mask], dim=-1)
        #cond_tensor = torch.cat([t_emb, u], dim=-1)

        res = self.bi_forward(input_tensor, edges, weights, t_emb)
        return res
