import torch
from torch import nn

from tsl.nn.layers import GraphConv, MultiHeadAttention

class TEncoder(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.channels  = channels

        self.inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        )

    def forward(self, t):
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
class TempModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = MultiHeadAttention(axis='time')

    def forward(self, v, q=None, k=None):
        if q is None and k is None:
            q = v
            k = v
        h = self.temporal_encoder(
            query=q,
            key=k,
            value=v
        )
        return h

class SpaModule(nn.Module):
    def __init__(self, is_pri=False):
        super().__init__()
        self.is_pri = is_pri
        self.spatial_encoder = MultiHeadAttention(axis='nodes')
        self.node_encoder = nn.Conv1d(1, 64, 1)
        self.gcn = GraphConv()
        
        self.norm_local = nn.GroupNorm(1, 64)
        self.norm_attn = nn.GroupNorm(1, 64)

        if not is_pri:
            self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
            )
            self.norm_final = nn.GroupNorm(1, 64)
        else:
            self.node_encoder_pri = nn.Conv1d(1, 64, 1)

    def forward_gcn(self, h, edges, nodes):
        h_gcn = self.gcn(h, edges, nodes) + h
        h_gcn = self.norm_local(h_gcn)

    def forward_attention(self, h, h_pri):
        v = self.node_encoder(h)
        if h_pri is not None:
            q = k = self.node_encoder(h_pri)
        else:
            q = k = v

        h_att = self.spatial_encoder(query=q,key=k,value=v) + h
        h_att = self.norm_attn(h_att)

    def forward_mlp(self, h_gcn, h_att):
        h_comb = h_gcn + h_att
        res = self.mlp(h_comb) + h_comb
        return self.norm_final(res)

    def forward(self, h, edges, nodes, h_pri=None):
        h_gcn = self.forward_gcn(h, edges, nodes)
        h_att = self.forward_attention(h, h_pri)

        return h_gcn, h_att if self.is_pri else self.forward_mlp(h_gcn, h_att)
            
class CFEM(nn.Module):
    '''Conditional Feature Extraction Module from priSTI'''
    def __init__(self):
        super().__init__()

        self.spa_module = SpaModule(is_pri=True)
        self.temp_module = TempModule()
        self.mlp = nn.Linear(64, 1)
        self.initial_conv = nn.Conv1d(1, 64, 1)
        self.norm = nn.GroupNorm(1, 64)

    def forward(self, x, edges, weights):
        h = self.initial_conv(x)
        h_temp = self.temp_module(h) + h
        
        h_gcn, h_att = self.spa_module(h_temp, edges, weights)

        h_final = h_temp + h_gcn + h_att

        h_prior = self.final_linear(h_final) + h_final
        return self.norm(h_prior)
    

class NEM(nn.Module):
    '''Noise Estimation Module from priSTI'''
    def __init__(self):
        super().__init__()
        self.temp_module = TempModule()
        self.spa_module = SpaModule()


        channels = None
        diffusion_embedding_dim = None
        self.cond_projection = nn.Conv1d(1, 2*channels, 1)
        self.mid_projection = nn.Conv1d(1, 2*channels, 1)
        self.output_projection = nn.Conv1d(1, 2*channels, 1)
        self.t_emb_compresser = nn.Linear(diffusion_embedding_dim, channels)

    def forward(self, h_in, h_pri, t_pos, edges, weights):

        t_pos_com = self.t_emb_compresser(t_pos) # Esto se queda en PriSTI (B, channels, 1)
        
        h_in_pos = h_in + t_pos_com

        h_tem = self.temp_module(q=h_pri, k=h_pri, v=h_in_pos)
        h_spa = self.spa_module(h_tem, edges, weights, h_pri) # Esto se queda en PriSTI (B, channels, K*L)
        h = self.mid_projection(h_spa) # Esto se queda en PriSTI (B, 2*channels, K*L)

        h_pri_projected = self.cond_projection(h_pri) # Esto se queda en PriSTI (B, 2*channels, K*L)
        h += h_pri_projected

        # aquí empieza a plicar multiples gates
        gate, filter_gate = torch.chunk(h_spa, 2, dim=-1)
        y = torch.sigmoid(gate) * torch.tanh(filter_gate)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=-1)
        residual = (h_in + residual)/torch.sqrt(2)

        return residual, skip
    
class priSTI(nn.Module):
    # Me falta por meterle el U
    def __init__(self):
        super().__init__()
        self.t_encoder = TEncoder()
        self.cfem = CFEM()

        self.input_projection = nn.Conv1d(1, 64, 1) # Genera features
        self.nem_layers = [NEM() for _ in range(4)]
        self.out = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 1)
        )

    def forward(self, x_ta_t, cond_info, t, edges, weights):
        # Curioso que no mete la máscara de imputación

        x_co = cond_info['x_co']
        h_pri = self.cfem(x_co, edges, weights)

        h_in = torch.cat([x_ta_t, x_co], dim=-1)
        t_pos = self.t_encoder(t)

        skip_connections = []
        for nem in self.nem_layers:
            h_in, skip = nem(h_in, h_pri, t_pos, edges, weights)
            skip_connections.append(skip)

        res = torch.cat(skip_connections, dim=-1)
        res = torch.sum(res, dim=-1) / torch.sqrt(len(self.nem_layers))

        return self.out(res)
    