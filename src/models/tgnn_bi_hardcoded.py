import torch
from torch import nn

from tsl.nn.blocks.decoders import GCNDecoder
from tsl.nn.blocks.encoders.conditional import ConditionalBlock

from src.utils import init_weights_xavier, clean_hyperparams, get_encoder, define_mlp_decoder


class TEncoder(nn.Module):
    def __init__(self, channels, time_dim=256,):
        super().__init__()
        self.channels  = channels
        self.time_dim = time_dim

        self.inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        )

    def forward(self, t):
        
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

class ConditionalEncoder(nn.Module):
    def __init__(self, hyperparameters=None, use_cond_info = True, output_size=None, time_shape=None):
        super().__init__()

        hyperparameters = {
            'encoder_name': 'tcn',
            'encoder': {
                'input_channels': 2,
                'hidden_channels': 64,
                'output_channels': 1,
                'kernel_size': 3,
                'dropout': 0.1,
                'n_layers': 4,
                'dilation': 1
            }
        }

        self.pos_encoding = TEncoder(256)
        self.output_size = 2
        self.use_cond_info = use_cond_info
        self.time_shape = time_shape
        self.output_size = output_size
        self.t_resampler = nn.Sequential(
                nn.Linear(256, output_size),
                nn.ReLU(),
            )

        if self.use_cond_info:
            self.cond_block = get_encoder(hyperparameters['encoder_name'])(**hyperparameters['encoder']).apply(init_weights_xavier)

            
    def forward(self, t, u):
        device = next(self.t_resampler.parameters()).device # Esto hay que revisarlo, lo he puesto para mover t a gpu
        t_emb = self.pos_encoding(t).to(device)
        cond_emb = self.t_resampler(t_emb).view(t.shape[0], 1, self.output_size, 1).repeat(1, self.time_shape, 1, 1) # Resample t to (B, T, N, F)
        if self.use_cond_info:
            u_emb = self.cond_block(u)
            cond_emb += u_emb
        return cond_emb

class UniModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

        self.name = hyperparameters['encoder_name']

        self.encoder = get_encoder(hyperparameters['encoder_name'])(**hyperparameters['encoder']).apply(init_weights_xavier)
        
        self.decoder = GCNDecoder(**hyperparameters['decoder']).apply(init_weights_xavier)

        self.cond_encoder = ConditionalEncoder(output_size=207, time_shape=24)

    def forward(self, x, t, cond_info, edges, weights):

        cond_embedding = self.cond_encoder(t, cond_info)

        x += cond_embedding
        x = self.encoder(x) if self.name != 'stcn' else self.encoder(x, edges, weights)

        x += cond_embedding
        x = self.decoder(x, edges, weights)
        return x

class BiModel(nn.Module):

    def __init__(self, args=None):
        super().__init__()
        args={
            'encoder_name': 'rnn',
            'encoder':{
                'input_size': 1,
                'hidden_size': 64,
                'output_size': 1,
                'exog_size': 0,
                'n_layers':3,
                'dropout':0.1,
                'cell':'gru'
            },
            'decoder':{
                'input_size': 1,
                'hidden_size': 64,
                'output_size': 1,
                'horizon': 24,
                'n_layers': 2,
                'dropout': 0.1
            }
        }
        self.model_f = UniModel(args)
        self.model_b = UniModel(args)
        self.cond_emb = ConditionalEncoder(output_size=207, time_shape=24)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
   
    def forward(self, x, t, cond_info, edges, weights):

        t = t.unsqueeze(-1).type(torch.float)

        
        f_representation = self.model_f(x, t, cond_info, edges, weights)
        b_representation = self.model_b(torch.flip(x, dims=[1]), t, torch.flip(cond_info, dims=[1]), edges, weights)

        cond_info = self.cond_emb(t, cond_info)

        h = torch.cat([
            f_representation + cond_info,
            b_representation + cond_info
            ], dim=-1)
        
        return self.decoder_mlp(h)
    