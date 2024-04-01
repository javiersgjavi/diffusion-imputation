import torch
import numpy as np
from torch import nn

from tsl.nn.layers import GraphConv, MultiHeadAttention, DiffConv, AdaptiveGraphConv, NodeEmbedding
from tsl.nn.layers.norm import LayerNorm
from src.utils import init_weights_xavier, init_weights_kaiming
from src.models.gcn import AdaptiveGCN

class SideInfo(nn.Module):
    def __init__(self, batch_size, time_steps, num_nodes):
        super().__init__()
        self.time_steps = time_steps
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        self.embed_layer = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=16)
        device = self.embed_layer.weight.device

        self.arange_nodes = torch.arange(self.num_nodes).to(device)
        self.time_embedding = self.get_time()

    def get_time(self):
        observed_tp = torch.arange(self.time_steps).unsqueeze(0)
        pos = torch.cat([observed_tp for _ in range(self.batch_size)], dim=0)
        self.div_term = 1 / torch.pow(
            10000.0, torch.arange(0, 128, 2) / 128
        )
        pe = torch.zeros(pos.shape[0], pos.shape[1], 128)
        position = pos.unsqueeze(2)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)

        pe = pe.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)

        return pe
    
    def get_node_info(self):
        self.arange_nodes = self.arange_nodes.to(self.embed_layer.weight.device)
        node_embed = self.embed_layer(self.arange_nodes)
        node_embed = node_embed.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.time_steps, -1, -1)
        return node_embed
    
    def forward(self, u):
        node_embed = self.get_node_info()
        self.time_embedding = self.time_embedding.to(u.device)
        side_info = torch.cat([self.time_embedding, node_embed], dim=-1)
        side_info = side_info[:u.shape[0]]
        return side_info

class TEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channels  = channels

        self.inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.channels, 2).float() / self.channels)
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
        ).apply(init_weights_xavier)

    def forward(self, t):
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq.to(t.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return self.mlp(pos_enc)
    
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

class SpaModule(nn.Module):
    def __init__(self, channels, heads, num_nodes=207, is_pri=False):
        super().__init__()
        self.is_pri = is_pri
        
        self.spatial_encoder = AttentionEncoded(channels, heads)
        self.gcn = AdaptiveGCN(channels)
        
        self.norm_local = nn.GroupNorm(4, channels).apply(init_weights_xavier)

        if not is_pri:
            self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.ReLU(),
            nn.Linear(channels*2, channels)
            ).apply(init_weights_xavier)
            self.norm_final = nn.GroupNorm(4, channels).apply(init_weights_xavier)
        else:
            self.node_encoder_pri = nn.Conv2d(num_nodes, channels, 1).apply(init_weights_kaiming)

    def forward_gcn(self, h, edges, weights, support):
        h_gcn = self.gcn(h, h.shape, support)
        h_gcn += h.reshape(h.shape[0], h.shape[-1], -1) # from (B, T, N, F) to (B, F, N*T)
        h_gcn = self.norm_local(h_gcn).view(h.shape) # from (B, F, N*T) to (B, T, N, F)
        return h_gcn

    def forward_mlp(self, h_gcn, h_att):
        h_comb = h_gcn + h_att
        res = self.mlp(h_comb) + h_comb
        res = res.view(res.shape[0], res.shape[3], -1) # from (B, T, N, F) to (B, F, T * N)
        return self.norm_final(res).view(h_comb.shape)

    def forward(self, h, edges, weights, h_pri=None, support=None):
        h_gcn = self.forward_gcn(h, edges, weights, support)
        h_att = self.spatial_encoder(h, h_pri)
        if self.is_pri:
            return h_gcn, h_att
        return self.forward_mlp(h_gcn, h_att)
            
class CFEM(nn.Module):
    '''Conditional Feature Extraction Module from priSTI'''
    def __init__(self, channels, heads, side_info_dim=144):
        super().__init__()

        self.initial_conv = nn.Conv2d(1, channels, 1).apply(init_weights_kaiming) # Creo que es cierto que este al final solo recibe itp
        self.initial_conv_side_info = nn.Conv2d(side_info_dim, channels, 1).apply(init_weights_kaiming)
        self.spa_module = SpaModule(channels, heads, is_pri=True)
        self.temp_module = TempModule(channels, heads)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.ReLU(),
            nn.Linear(channels*2, channels)
        ).apply(init_weights_xavier)
        self.norm = nn.GroupNorm(4, channels).apply(init_weights_xavier)

    def forward(self, x, edges, weights, side_info, support):
        x = self.initial_conv(x.permute(0,3,1,2)) # from (B, T, N, 1) to (B, 1, T, N)
        side_info = self.initial_conv_side_info(side_info.permute(0,3,1,2)) # from (B, T, N, 1) to (B, 1, T, N)
        h = x + side_info
        h = h.permute(0,2,3,1) # from (B, F, T, N) to (B, T, N, F)
        h_temp = self.temp_module(h) + h
        h_gcn, h_att = self.spa_module(h_temp, edges, weights, support=support)

        h_final = h_temp + h_gcn + h_att

        h_prior = self.mlp(h_final) + h_final
        h_prior = h_prior.view(h_prior.shape[0], h_prior.shape[3], -1) # from (B, T, N, F) to (B, F, T * N)
        h_prior = self.norm(h_prior).view(h_final.shape) # from (B, F, T * N) to (B, T, N, F)
        return nn.functional.relu(h_prior)
    

class NEM(nn.Module):
    '''Noise Estimation Module from priSTI'''
    def __init__(self, channels, heads, t_emb_size, side_info_dim=144):
        super().__init__()
        self.temp_module = TempModule(channels, heads)
        self.spa_module = SpaModule(channels, heads, is_pri=False)

        self.cond_projection = nn.Conv2d(side_info_dim, 2*channels, 1).apply(init_weights_kaiming)
        self.mid_projection = nn.Conv2d(channels, 2*channels, 1).apply(init_weights_kaiming)
        self.output_projection = nn.Conv2d(channels, 2*channels, 1).apply(init_weights_kaiming)
        self.t_emb_compresser = nn.Linear(t_emb_size, channels).apply(init_weights_xavier)
        

    def forward(self, h_in, h_pri, t_pos, side_info, edges, weights, support):

        t_pos_com = self.t_emb_compresser(t_pos) # Esto se queda en PriSTI (B, channels, 1)
        h_in_pos = h_in + t_pos_com

        h_tem = self.temp_module(q=h_pri, k=h_pri, v=h_in_pos)
        h_spa = self.spa_module(h_tem, edges, weights, h_pri, support=support) # Esto se queda en PriSTI (B, T, N, F)
        h_spa = h_spa.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        h = self.mid_projection(h_spa) # (B, 2*F, T, N)

        h_side_info = self.cond_projection(side_info.permute(0, 3, 1, 2)) # (B, 2*F, T, N)
        h += h_side_info # (B, 2*F, T, N)
        #h = h.permute(0, 2, 3, 1) # from (B, 2*F, T, N) to (B, T, N, 2*F)

        # aquí empieza aplicar multiples gates
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
        self.side_info = SideInfo(
            batch_size=4, 
            time_steps=24, 
            num_nodes=207)
        self.t_encoder = TEncoder(t_emb_size)
        self.cfem = CFEM(channels, heads)

        self.input_projection = nn.Sequential(
            nn.Conv2d(2, channels, 1),
            nn.ReLU()
        ).apply(init_weights_kaiming)

        self.nem_layers = nn.ModuleList([NEM(channels, heads, t_emb_size) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1)
        ).apply(init_weights_kaiming)


        self.support = torch.load('support.pt')
        self.support[0] = self.support[0]
        self.support[1] = self.support[1]
        self.nodevec1 = nn.Parameter(torch.randn(207, 10), requires_grad=True) # Esto tengo que sacarlo de aqui
        self.nodevec2 = nn.Parameter(torch.randn(10, 207), requires_grad=True)
        self.support.append([self.nodevec1, self.nodevec2])

    def forward(self, x_ta_t, x_co, u, t, edges, weights):

        side_info = self.side_info(u)

        #mask = cond_info['mask_co']
        h_in = torch.cat([x_ta_t, x_co], dim=-1) # (B, T, N, 2)
        h_in = h_in.permute(0, 3, 1, 2) # from (B, T, N, 2) to (B, 2, T, N)
        h_in = self.input_projection(h_in).permute(0, 2, 3, 1) # from (B, 2, T, N) to (B, T, N, F)
        
        h_pri = self.cfem(x_co, edges, weights, side_info, self.support)

        t = t.unsqueeze(-1).type(torch.float)
        t_pos = self.t_encoder(t).unsqueeze(1).unsqueeze(1).repeat((1, h_in.shape[1], h_in.shape[2], 1)) # Resample t to (B, T, N, F)

        skip_connections = []
        for nem in self.nem_layers:
            h_in, skip = nem(h_in, h_pri, t_pos, side_info, edges, weights, self.support)
            skip_connections.append(skip)

        res = torch.stack(skip_connections)
        len_layers = torch.tensor(len(self.nem_layers)).to(res.device)
        res = torch.sum(res, dim=0) / torch.sqrt(len_layers)

        res = res.permute(0, 3, 1, 2) # from (B, T, N, F) to (B, F, T, N)
        res = self.out(res) # (B, 1, T, N)
        res = res.permute(0, 2, 3, 1) # from (B, 1, T, N) to (B, T, N, 1)
        return res
    