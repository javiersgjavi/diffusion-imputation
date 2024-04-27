import torch
import torch.nn as nn

from mamba_ssm import Mamba
from tsl.nn.layers import MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm

from einops import rearrange
from src.utils import init_weights_xavier, init_weights_kaiming, _init_weights_mamba

from einops.layers.torch import Rearrange
# from VMamba.classification.models.vmamba import VSSBlock

from Vim.mamba.mamba_ssm.modules.mamba_simple import Mamba as BiMamba

class WrapperMambaModule(nn.Module):
    def __init__(self, is_pri=False, t=24, n=207, sum=True):
        super().__init__()
        self.is_pri = is_pri
        self.t = t
        self.n = n
        self.sum = sum

    def reshape_pri(self, x):
        t, bn, f = x.shape
        b = bn//self.n
        x = x.reshape(t, b, self.n, f)
        x = x.permute(0,2,1,3).reshape(b*self.n, self.t, f)
        return x

    def reshape_nem(self, x, qk):
        b, f, n, t = qk.shape
        x = x.reshape(b, f, n, t).permute(0, 2, 3, 1).reshape(b*n, t, f)
        qk = qk.permute(0, 2, 3, 1).reshape(b*n, t, f)
        x = x + qk if self.sum else torch.cat([x, qk], dim=-1)
        return x

    def reshape_out(self, x):
        if not self.is_pri:
            bn, t, f = x.shape
            b = bn//self.n
            x = x.reshape(b, self.n, t, f)
            x = x.permute(0, 3, 1, 2).reshape(b, f, self.n*t)
        return x
    
    def reshape_in(self, x, qk=None):
        if self.is_pri:
            x = self.reshape_pri(x)
        elif not self.is_pri:
            x = self.reshape_nem(x, qk)
        return x
    
class CustomMamba(WrapperMambaModule):
    def __init__(self, channels, dropout=0.1, is_pri=False, t=24, n=207):
        super().__init__(is_pri, t, n)

        self.block = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(channels) if not is_pri else nn.Identity(),
            #Mamba(d_model=channels).apply(_init_weights_mamba),
            BiMamba(d_model=channels, bimamba_type='v2').apply(_init_weights_mamba),
            nn.LayerNorm(channels),
        ).apply(_init_weights_mamba)

    def forward(self, x, qk=None):
        x = self.reshape_in(x, qk)
        h = self.block(x)
        if self.is_pri:
            h +=  x
        h = self.reshape_out(h)
        return h

class CustomBiMamba(WrapperMambaModule):
    def __init__(self, channels, dropout=0.1, is_pri=False, t=24, n=207):
        super().__init__(is_pri, t, n, sum=False)

        self.input = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels*2, channels) if not is_pri else nn.Identity(),
        )

        self.block_f = Mamba(d_model=channels).apply(_init_weights_mamba).apply(_init_weights_mamba)

        self.block_b =  Mamba(d_model=channels).apply(_init_weights_mamba).apply(_init_weights_mamba)

        self.norm = nn.LayerNorm(channels)


    def forward(self, x, qk=None):
        x = self.reshape_in(x, qk)
        h = self.input(x)

        h_fw = self.block_f(h)
        h_bw = self.block_b(h.flip(1)).flip(1)

        h = h_fw + h_bw

        h =  self.norm(h)

        '''if self.is_pri:
            h +=  x'''
        h = self.reshape_out(h)
        return h
    
class MambaDualScan(nn.Module):
    def __init__(self, channels, axis='time', dropout=0.1, is_pri=False):
        super().__init__()

        input_size = 2*channels if not is_pri else channels
        reducted_size = channels//2

        self._in_pattern = f'b t n f -> (b n) t f'
        self._out_pattern = f'(b n) t f -> b t n f'

        self.input = nn.Sequential(
            Rearrange(self._in_pattern, t=24, n=207),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, reducted_size),
        ).apply(init_weights_xavier)

        self.mamba_fw = Mamba(d_model=reducted_size).apply(_init_weights_mamba)
        self.mamba_bw = Mamba(d_model=reducted_size).apply(_init_weights_mamba)

        self.output = nn.Sequential(
            LayerNorm(reducted_size),
            nn.Linear(reducted_size, channels),
            nn.Dropout(dropout),
        ).apply(init_weights_xavier)
        
    def forward(self, x, qk=None):
        
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)

        h = self.input(x)

        h_fw = self.mamba_fw(h)
        h_bw = self.mamba_bw(h.flip(1)).flip(1)

        h = h_fw + h_bw

        return self.output(h)
    
class ViMamba(nn.Module):
    def __init__(self, channels, dropout=0.1, is_pri=False):
        super().__init__()

        input_size = 2*channels if not is_pri else channels
        reducted_size = channels//2

        self._in_pattern = f'b t n f -> (b t) n f'
        self._out_pattern = f'(b t) n  f -> b t n f'

        self.input = nn.Sequential(
            Rearrange(self._in_pattern, t=24, n=207),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, reducted_size),
        ).apply(init_weights_xavier)
        
        self.mamba_fw = Mamba(d_model=reducted_size).apply(_init_weights_mamba)
        self.mamba_bw = Mamba(d_model=reducted_size).apply(_init_weights_mamba)

        self.output_vim = nn.Sequential(
            LayerNorm(reducted_size),
            nn.Linear(reducted_size, channels),
            nn.Dropout(dropout),
            Rearrange(self._out_pattern, t=24, n=207),
        ).apply(init_weights_xavier)

        self.output = nn.Sequential(
            LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        ).apply(init_weights_xavier)


    def forward(self, x, qk=None):
        x_in = x
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)

        h = self.input(x)

        h_fw = self.mamba_fw(h)
        h_bw = self.mamba_bw(h.flip(1)).flip(1)

        h = h_fw + h_bw

        h = self.output_vim(h) + x_in

        return self.output(h)

class MambaNodeMamba(nn.Module):
    def __init__(self, channels, dropout=0.1, is_pri=False):
        super().__init__()

        input_size = 2*channels if not is_pri else channels
        reducted_size = channels//2


        self._in_pattern = f'b t n f -> b (n t) f'
        self._out_pattern = f'b (n t) f -> b t n f'

        self.block = nn.Sequential(
            Rearrange(self._in_pattern, t=24, n=207),
            nn.Linear(input_size, channels).apply(init_weights_xavier),
            Mamba(d_model=channels).apply(_init_weights_mamba),
            nn.Dropout(dropout),
            Rearrange(self._out_pattern, t=24, n=207),
        )

        self.norm = nn.Sequential(
            Rearrange('b t n f -> b f t n'),
            nn.GroupNorm(4, channels),
            Rearrange('b f t n -> b t n f'),
        )

    def forward(self, x, qk=None):
        x_in = x
        if qk is not None:
            x = torch.cat([x, qk], dim=-1)

        h = self.block(x) + x_in

        return self.norm(h)


class MambaNode(nn.Module):
    def __init__(self, channels, dropout=0.1, is_pri=False):
        super().__init__()

        # self.vmamba = VSSBlock(hidden_dim=channels//2, forward_type='v0')
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