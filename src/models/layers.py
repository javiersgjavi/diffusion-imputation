import torch
import torch.nn as nn

from mamba_ssm import Mamba
from tsl.nn.layers import MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm

from einops import rearrange
from src.utils import init_weights_xavier, init_weights_kaiming

class CustomMamba(nn.Module):
    def __init__(self, channels, axis='time'):
        super().__init__()
        print(channels)
        self.mamba = Mamba(
            d_model=channels,
        )

        if axis == 'time':
            shape = '(b n) t f'
        elif axis == 'nodes':
            shape = '(b t) n f'

        self._in_pattern = f'b t n f -> {shape}'
        self._out_pattern = f'{shape} -> b t n f'

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, self._in_pattern)
        x = self.mamba(x)
        return rearrange(x, self._out_pattern, b=b)


class MambaTime(nn.Module):
    def __init__(self, channels, dropout=0.1, is_pri=False, axis='time'):
        super().__init__()
        self.mamba = CustomMamba(channels, axis)
        self.dropout = nn.Dropout(dropout)

        if not is_pri:
            self.layer_norm = LayerNorm(channels)
        
    def forward(self, v, qk=None):
        if qk is not None:
            v += qk
            v = self.layer_norm(v)

        h = self.mamba(v)
        return self.dropout(h)
    
class TransformerTime(nn.Module):
    def __init__(self, channels, heads, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.temporal_encoder = MultiHeadAttention(axis='time', embed_dim=channels, heads=heads, dropout=dropout).apply(init_weights_xavier)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(channels)
        self.first_mlp_layer = nn.Sequential(
            LayerNorm(channels),
            nn.Linear(channels, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, channels),
            nn.Dropout(dropout)
        ).apply(init_weights_xavier)

    def forward(self, v, qk=None):
        if qk is None:
            qk = v
        h_att = self.temporal_encoder(query=qk,key=qk,value=v)[0]
        h = v + self.dropout(h_att)
        h_mlp = h + self.first_mlp_layer(h)
        return self.layer_norm(h_mlp)

class AttentionEncoded(nn.Module):
    def __init__(self, channels=64, heads=8, num_nodes=207):
        super().__init__()

        self.channels = channels
        self.heads = heads

        self.node_encoder = nn.Conv2d(num_nodes, self.channels, 1, bias=False).apply(init_weights_kaiming)
        self.node_encoder_pri = nn.Conv2d(num_nodes, self.channels, 1, bias=False).apply(init_weights_kaiming)
        self.node_decoder = nn.Conv2d(self.channels, num_nodes, 1, bias=False).apply(init_weights_kaiming)

        self.spatial_encoder = MultiHeadAttention(axis='nodes', embed_dim=self.channels, heads=self.heads, dropout=0.1).apply(init_weights_xavier)

        self.norm_attn = nn.GroupNorm(4, self.channels).apply(init_weights_xavier)

    def get_qkv(self, h, h_pri=None):
        h = h.permute(0, 2, 1, 3) # from (B, T, N, F) to (B, N, T, F)
        v = self.node_encoder(h).permute(0, 2, 1, 3) # from (B, N, T, F) to (B, T, K, F)
        if h_pri is not None:
           h_pri = h_pri.permute(0, 2, 1, 3) # from (B, T, N, F) to (B, N, T, F)
           q = k = self.node_encoder_pri(h_pri).permute(0, 2, 1, 3)
        else:
            q = k = v
        return q, k, v
    
    def forward(self, h, h_pri):
        q, k, v = self.get_qkv(h, h_pri)

        h_att = self.spatial_encoder(query=q,key=k,value=v)[0]
        h_att = h_att.permute(0, 2, 1, 3) # from (B, T, K, F) to (B, K, T, F)
        h_att = self.node_decoder(h_att).permute(0, 2, 1, 3) + h # from (B, K, T, F) to (B, T, N, F)
        h_att = h_att.reshape(h.shape[0], h.shape[3], -1) # from (B, T, N, F) to (B, F, N*T)
        h_att = self.norm_attn(h_att).view(h.shape) # from (B, F, N*T) to (B, T, N, F)
        return h_att