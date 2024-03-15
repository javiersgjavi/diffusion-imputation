import torch
from torch import nn

from tsl.nn.layers import GraphConv, MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm
from src.utils import init_weights_xavier

class TEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channels  = channels

        self.inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.channels, 2).float() / self.channels)
        )

    def forward(self, t):
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
class TempModule(nn.Module):
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


    def forward(self, v, q=None, k=None):
        if q is None or k is None:
            q = k= v
        h_att = self.temporal_encoder(query=q,key=k,value=v)[0]
        h = v + self.dropout(h_att)
        h_mlp = h + self.first_mlp_layer(h)
        return self.layer_norm(h_mlp)

class SpaModule(nn.Module):
    def __init__(self, channels, heads, num_nodes=207, is_pri=False):
        super().__init__()
        self.is_pri = is_pri
        self.node_encoder = nn.Conv2d(num_nodes, channels, 1).apply(init_weights_xavier)
        self.node_encoder_pri = nn.Conv2d(num_nodes, channels, 1).apply(init_weights_xavier)
        self.node_decoder = nn.Conv2d(channels, num_nodes, 1).apply(init_weights_xavier)
        self.spatial_encoder = MultiHeadAttention(axis='nodes', embed_dim=channels, heads=heads, dropout=0.1).apply(init_weights_xavier)
        self.gcn = GraphConv(input_size=channels, output_size=channels).apply(init_weights_xavier)
        
        self.norm_local = nn.GroupNorm(4, channels).apply(init_weights_xavier)
        self.norm_attn = nn.GroupNorm(4, channels).apply(init_weights_xavier)

        if not is_pri:
            self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.ReLU(),
            nn.Linear(channels*2, channels)
            ).apply(init_weights_xavier)
            self.norm_final = nn.GroupNorm(4, channels).apply(init_weights_xavier)
        else:
            self.node_encoder_pri = nn.Conv2d(num_nodes, channels, 1).apply(init_weights_xavier)

    def forward_gcn(self, h, edges, weights):
        h_gcn = self.gcn(h, edges, weights) + h
        h_gcn = h_gcn.view(h_gcn.shape[0], h_gcn.shape[3], -1) # from (B, T, N, F) to (B, F, N*T)
        h_gcn = self.norm_local(h_gcn).view(h.shape) # from (B, F, N*T) to (B, T, N, F)
        return h_gcn

    def forward_attention(self, h, h_pri):
        v = self.node_encoder(h.permute(0, 2, 1, 3)) # from (B, T, N, F) to (B, K, T, F)
        v = v.permute(0, 2, 3, 1) # from (B, K, T, F) to (B, T, K, F)
        if h_pri is not None:
            h_pri = h_pri.permute(0, 2, 1, 3) # from (B, T, N, F) to (B, N, T, F)
            h_pri = self.node_encoder_pri(h_pri) # from (B, N, T, F) to (B, K, T, F)
            q = k = h_pri.permute(0, 2, 1, 3) # from (B, K, T, F) to (B, T, K, F)
        else:
            q = k = v

        h_att = self.spatial_encoder(query=q,key=k,value=v)[0]
        h_att = h_att.permute(0, 2, 1, 3) # from (B, T, K, F) to (B, K, T, F)
        h_att = self.node_decoder(h_att).permute(0, 2, 1, 3) + h # from (B, K, T, F) to (B, T, N, F)
        h_att = h_att.reshape(h.shape[0], h.shape[3], -1) # from (B, T, N, F) to (B, F, N*T)
        h_att = self.norm_attn(h_att).view(h.shape) # from (B, F, N*T) to (B, T, N, F)
        return h_att

    def forward_mlp(self, h_gcn, h_att):
        h_comb = h_gcn + h_att
        res = self.mlp(h_comb) + h_comb
        res = res.view(res.shape[0], res.shape[3], -1) # from (B, T, N, F) to (B, F, T * N)
        return self.norm_final(res).view(h_comb.shape)

    def forward(self, h, edges, weights, h_pri=None):
        h_gcn = self.forward_gcn(h, edges, weights)
        h_att = self.forward_attention(h, h_pri)
        if self.is_pri:
            return h_gcn, h_att
        return self.forward_mlp(h_gcn, h_att)
            
class CFEM(nn.Module):
    '''Conditional Feature Extraction Module from priSTI'''
    def __init__(self, channels, heads):
        super().__init__()

        self.initial_conv = nn.Conv2d(3, channels, 1).apply(init_weights_xavier)
        self.initial_conv_u = nn.Conv2d(2, channels, 1).apply(init_weights_xavier)
        self.spa_module = SpaModule(channels, heads, is_pri=True)
        self.temp_module = TempModule(channels, heads)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.ReLU(),
            nn.Linear(channels*2, channels)
        ).apply(init_weights_xavier)
        self.norm = nn.GroupNorm(4, channels).apply(init_weights_xavier)

    def forward(self, x, edges, weights, u):
        h = self.initial_conv(x.permute(0,3,1,2)) + self.initial_conv_u(u.permute(0,3,1,2)) # from (B, T, N, 1) to (B, 1, T, N)
        h = h.permute(0,2,3,1) # from (B, F, T, N) to (B, T, N, F)
        h_temp = self.temp_module(h) + h
        
        h_gcn, h_att = self.spa_module(h_temp, edges, weights)

        h_final = h_temp + h_gcn + h_att

        h_prior = self.mlp(h_final) + h_final
        h_prior = h_prior.view(h_prior.shape[0], h_prior.shape[3], -1) # from (B, T, N, F) to (B, F, T * N)
        h_prior = self.norm(h_prior).view(h_final.shape) # from (B, F, T * N) to (B, T, N, F)
        return nn.functional.relu(h_prior)
    

class NEM(nn.Module):
    '''Noise Estimation Module from priSTI'''
    def __init__(self, channels, heads, t_emb_size):
        super().__init__()
        self.temp_module = TempModule(channels, heads)
        self.spa_module = SpaModule(channels, heads, is_pri=False)

        self.cond_projection = nn.Conv2d(channels, 2*channels, 1).apply(init_weights_xavier)
        self.mid_projection = nn.Conv2d(channels, 2*channels, 1).apply(init_weights_xavier)
        self.output_projection = nn.Conv2d(channels, 2*channels, 1).apply(init_weights_xavier)
        self.t_emb_compresser = nn.Sequential(
            nn.Linear(t_emb_size, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
        ).apply(init_weights_xavier)
        

    def forward(self, h_in, h_pri, t_pos, edges, weights):

        t_pos_com = self.t_emb_compresser(t_pos) # Esto se queda en PriSTI (B, channels, 1)
        h_in_pos = h_in + t_pos_com

        h_tem = self.temp_module(q=h_pri, k=h_pri, v=h_in_pos)
        h_spa = self.spa_module(h_tem, edges, weights, h_pri) # Esto se queda en PriSTI (B, T, N, F)
        h_spa = h_spa.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        h = self.mid_projection(h_spa) # (B, 2*F, T, N)

        h_pri = h_pri.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        h_pri_projected = self.cond_projection(h_pri) # (B, 2*F, T, N)
        h += h_pri_projected # (B, 2*F, T, N)
        #h = h.permute(0, 2, 3, 1) # from (B, 2*F, T, N) to (B, T, N, 2*F)

        # aquí empieza a plicar multiples gates
        gate, filter_gate = torch.chunk(h, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_gate)
        #y = y.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        y = self.output_projection(y) # (B, 2*F, T, N)
        #y = y.permute(0, 2, 3, 1) # from (B, 2*F, T, N) to (B, T, N, 2*F)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.permute(0, 2, 3, 1) # from (B, F, T, N) to (B, T, N, F)
        skip = skip.permute(0, 2, 3, 1) # from (B, F, T, N) to (B, T, N, F)
        two = torch.tensor(2.0).to(residual.device)
        residual = (h_in + residual)/torch.sqrt(two)

        return residual, skip
    
class PriSTI(nn.Module):
    # Me falta por meterle el U
    def __init__(self, channels=64, heads=8, t_emb_size=128):
        super().__init__()
        self.t_encoder = TEncoder(t_emb_size)
        self.cfem = CFEM(channels, heads)

        self.input_projection = nn.Conv2d(3, channels, 1).apply(init_weights_xavier) # Genera features
        self.nem_layers = nn.ModuleList([NEM(channels, heads, t_emb_size) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1)
        ).apply(init_weights_xavier)

    def forward(self, x_ta_t, cond_info, t, edges, weights):
        # Curioso que no mete la máscara de imputación

        u = cond_info['u']
        x_co = cond_info['x_co']
        mask = cond_info['mask_co']
        h_in = torch.cat([x_ta_t, x_co, mask], dim=-1) # (B, T, N, 2)
        
        h_pri = self.cfem(h_in, edges, weights, u)

        h_in = h_in.permute(0, 3, 1, 2) # from (B, T, N, 2) to (B, 2, T, N)
        h_in = self.input_projection(h_in).permute(0, 2, 3, 1) # from (B, 2, T, N) to (B, T, N, F)
        
        t = t.unsqueeze(-1).type(torch.float)
        t_pos = self.t_encoder(t).unsqueeze(1).unsqueeze(1).repeat((1, h_in.shape[1], h_in.shape[2], 1)) # Resample t to (B, T, N, F)

        skip_connections = []
        for nem in self.nem_layers:
            h_in, skip = nem(h_in, h_pri, t_pos, edges, weights)
            skip_connections.append(skip)

        res = torch.stack(skip_connections)
        len_layers = torch.tensor(len(self.nem_layers)).to(res.device)
        res = torch.sum(res, dim=0) / torch.sqrt(len_layers)

        res = res.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        res = self.out(res) # (B, 1, T, N)
        res = res.permute(0, 2, 3, 1) # from (B, 1, T, N) to (B, T, N, 1)
        return res
    