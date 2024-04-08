import torch
import torch.nn as nn

from mamba_ssm import Mamba
from tsl.nn.layers import MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm

from einops import rearrange
from src.utils import init_weights_xavier, init_weights_kaiming

from einops.layers.torch import Rearrange
from VMamba.classification.models.vmamba import VSSBlock

class CustomMambaDualScan(nn.Module):
    def __init__(self, channels, axis='time', dropout=0.1, is_pri=False):
        super().__init__()

        print(is_pri)
        input_size = 2*channels if not is_pri else channels
        reducted_size = channels//2

        self.input_layer = nn.Linear(input_size, reducted_size).apply(init_weights_xavier)
        self.mamba_fw = Mamba(
            d_model=reducted_size,
        )

        self.mamba_bw = Mamba(
            d_model=reducted_size,
        )

        self.layer_norm = LayerNorm(reducted_size)
        self.output = nn.Linear(reducted_size, channels).apply(init_weights_xavier)
        self.dropout = nn.Dropout(dropout)
        

        shape = '(b n) t f'

        self._in_pattern = f'b t n f -> {shape}'
        self._out_pattern = f'{shape} -> b t n f'

    def forward(self, x, qk=None):
        b, t, n, f = x.shape
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)

        x = self.input_layer(x)
        x = rearrange(x, self._in_pattern, b=b, t=t)

        x_fw = self.mamba_fw(x)
        x_bw = self.mamba_bw(x.flip(1)).flip(1)

        x = x_fw + x_bw

        x = rearrange(x, self._out_pattern, b=b, t=t)

        x = self.layer_norm(x)

        x = self.output(x)

        return self.dropout(x)

class CustomMamba(nn.Module):
    def __init__(self, channels, axis='time', dropout=0.1, is_pri=False):
        super().__init__()

        self.axis = axis

        self.mamba = Mamba(
            d_model=channels,
        )
        self.dropout = nn.Dropout(dropout)
        
        if axis == 'time':
            shape = '(b n) t f'
        elif axis == 'nodes':
            shape = 'b (t n) f'

        self._in_pattern = f'b t n f -> {shape}'
        self._out_pattern = f'{shape} -> b t n f'

        if axis == 'nodes':
            self.layer_norm = LayerNorm(channels)

        if not is_pri:
            self.info_mixer = nn.Linear(channels*2, channels).apply(init_weights_xavier)

    def forward(self, x, qk=None):
        b = x.shape[0]
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)
            x = self.info_mixer(x)

        x = rearrange(x, self._in_pattern, b=b, t=24)
        if self.axis == 'nodes':
            x = self.layer_norm(x)
        x = self.mamba(x)
        x = rearrange(x, self._out_pattern, b=b, t=24)
        return self.dropout(x)

class MambaTime(CustomMambaDualScan):
    def __init__(self, *args, **kwargs):
        super().__init__(axis='time',*args, **kwargs)

class MambaNode(nn.Module):
    def __init__(self, channels, dropout=0.1, is_pri=False):
        super().__init__()

        self.vmamba = VSSBlock(hidden_dim=channels//2, forward_type='v0')
        self.dropout = nn.Dropout(dropout)

        if is_pri:
            self.input_layer = nn.Linear(channels, channels//2).apply(init_weights_xavier)
        else:
            self.info_mixer = nn.Linear(channels*2, channels//2).apply(init_weights_xavier)
        
        self.output = nn.Linear(channels//2, channels).apply(init_weights_xavier)

    def forward(self, x, qk=None):
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)
            x = self.info_mixer(x)
        else:
            x = self.input_layer(x)

        x = self.vmamba(x)
        return self.output(self.dropout(x))

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
        # Esto está mal implementado, las capas mlp no sirven para nada, el temporal encoder ya lo hace todo
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
    
class Conv2DCustom(nn.Module):
    def __init__(self, in_channels, out_channels, reorder_out=True, reorder_in=True):
        super().__init__()

        self.layers = nn.Sequential(
            Rearrange('b t n f -> b f t n') if reorder_in else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, 1),
            Rearrange('b f t n -> b t n f') if reorder_out else nn.Identity()
        ).apply(init_weights_kaiming)

    def forward(self, x):
        return self.layers(x)