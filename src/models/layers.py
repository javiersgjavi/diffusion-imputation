import torch
import torch.nn as nn

from mamba_ssm import Mamba, Mamba2
from src.utils import _init_weights_mamba


from src.models.bimamba.modules.mamba_simple import BiMamba
from src.models.bimamba.modules.mamba2_simple import BiMamba2

class WrapperMambaModule(nn.Module):
    def __init__(self, is_pri=False, t=24, n=207):
        super().__init__()
        self.is_pri = is_pri
        self.t = t
        self.n = n

    def reshape_pri(self, x):
        t, bn, f = x.shape
        b = bn//self.n
        x = x.reshape(t, b, self.n, f)
        x = x.permute(0,2,1,3).reshape(b*self.n, self.t, f)
        qk = torch.zeros_like(x)
        return x, qk

    def reshape_nem(self, x, qk):
        b, f, n, t = qk.shape
        x = x.reshape(b, f, n, t).permute(0, 2, 3, 1).reshape(b*n, t, f)
        qk = qk.permute(0, 2, 3, 1).reshape(b*n, t, f)
        return x, qk

    def reshape_out(self, x):
        if not self.is_pri:
            bn, t, f = x.shape
            b = bn//self.n
            x = x.reshape(b, self.n, t, f)
            x = x.permute(0, 3, 1, 2).reshape(b, f, self.n*t)
        return x
    
    def reshape_in(self, x, qk=None):
        if self.is_pri:
            x, qk = self.reshape_pri(x)
        elif not self.is_pri:
            x, qk = self.reshape_nem(x, qk)
        return x, qk
    
class CustomMamba(WrapperMambaModule):
    def __init__(self, channels, dropout=0.1, is_pri=False, t=24, n=207, bidirectional=True, n_blocks=2):
        super().__init__(is_pri, t, n)


        mamba_block = BiMamba(d_model=channels, bimamba_type='v2') if bidirectional else Mamba(d_model=channels)
        # mamba_block = Mamba2(d_model=channels, headdim=128//4, use_mem_eff_path=False, expand=4)
        # mamba_block = Mamba(d_model=channels)
        # mamba_block = BiMamba2(d_model=channels, headdim=128//4, use_mem_eff_path=False, expand=4)
        # mamba_block = BiMamba2(d_model=channels, headdim=128//8, use_mem_eff_path=False, expand=2)


        self.block = nn.Sequential(
            nn.LayerNorm(channels) if not is_pri else nn.Identity(),
            nn.Dropout(dropout),
            mamba_block.apply(_init_weights_mamba),
            nn.LayerNorm(channels),
        ).apply(_init_weights_mamba)


        self.sub_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Dropout(dropout),
                    mamba_block.apply(_init_weights_mamba),
                    nn.LayerNorm(channels)
                    ).apply(_init_weights_mamba)
                for _ in range(n_blocks-1)
                ]
            )

        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x, qk=None):
        x, qk = self.reshape_in(x, qk)
        h = x + qk

        h = self.block(h) + h

        for sub_block in self.sub_blocks:
            h = sub_block(h) + h

        h = self.layer_norm(h)
        h = self.reshape_out(h)
        return h
