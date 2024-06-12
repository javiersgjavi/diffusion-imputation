import torch
import torch.nn as nn

from mamba_ssm import Mamba
from src.utils import _init_weights_mamba


from src.models.bimamba.modules.mamba_simple import BiMamba

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
        x_sum = x + qk if self.sum else torch.cat([x, qk], dim=-1)
        return x_sum

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
    def __init__(self, channels, dropout=0.1, is_pri=False, t=24, n=207, bidirectional=True):
        super().__init__(is_pri, t, n)

        mamba_block = BiMamba(d_model=channels, bimamba_type='v2') if bidirectional else Mamba(d_model=channels)

        self.block = nn.Sequential(
            nn.LayerNorm(channels) if not is_pri else nn.Identity(),
            nn.Dropout(dropout),
            mamba_block.apply(_init_weights_mamba),
            nn.LayerNorm(channels),
        ).apply(_init_weights_mamba)

        self.layer_norm = nn.LayerNorm(channels)

    
    def forward(self, x, qk=None):
        x = self.reshape_in(x, qk)
        h = self.block(x)
        h = self.layer_norm(h+x)
        h = self.reshape_out(h)
        return h
