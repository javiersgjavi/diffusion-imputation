import torch
from torch import nn
from einops import rearrange

from tsl.nn.layers import GraphConv, MultiHeadAttention, DiffConv, AdaptiveGraphConv, NodeEmbedding
from src.utils import init_weights_xavier, init_weights_kaiming
from src.models.gcn import AdaptiveGCN

from src.models.layers import TransformerTime, AttentionEncoded, MambaTime, Conv2DCustom, MambaNode
from einops.layers.torch import Rearrange

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
    def __init__(self, channels, heads, is_pri=False):
        super().__init__()

        self.layer = MambaTime(channels, is_pri=is_pri)
        
        self.group_norm = nn.Sequential(
            Rearrange('b t n f -> b f t n'),
            nn.GroupNorm(4, channels),
            Rearrange('b f t n -> b t n f')
        ).apply(init_weights_xavier)

    def forward(self, v , qk=None):
        v = self.layer(v, qk) + v
        return self.group_norm(v)

class SpaModule(nn.Module):
    def __init__(self, channels, heads, num_nodes=207, is_pri=False):
        super().__init__()
        self.is_pri = is_pri
        
        self.spatial_encoder = AttentionEncoded(channels, heads)
        self.gcn = AdaptiveGCN(channels)
        
        self.norm_local = nn.Sequential(
            nn.GroupNorm(4, channels),
            Rearrange('b f (n t) -> b t n f', t=24)
        ).apply(init_weights_xavier)

        if not is_pri:
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels*2),
                nn.ReLU(),
                nn.Linear(channels*2, channels)
            ).apply(init_weights_xavier)

            self.norm_final = nn.Sequential(
                Rearrange('b t n f-> b f (t n)'),
                nn.GroupNorm(4, channels),
                Rearrange('b f (t n) -> b t n f', t=24)
                ).apply(init_weights_xavier)
        else:
            self.node_encoder_pri = nn.Conv2d(num_nodes, channels, 1).apply(init_weights_kaiming)

    def forward_gcn(self, h, edges, weights, support):
        h_gcn = self.gcn(h, h.shape, support)
        h_gcn += rearrange(h, 'b t n f -> b f (n t)')
        return self.norm_local(h_gcn)

    def forward_mlp(self, h_gcn, h_att):
        h_comb = h_gcn + h_att
        res = self.mlp(h_comb) + h_comb
        return self.norm_final(res)

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

        self.initial_conv = Conv2DCustom(1, channels, reorder_out=False)
        self.initial_conv_side_info = Conv2DCustom(side_info_dim, channels, reorder_out=False)
        
        self.spa_module = SpaModule(channels, heads, is_pri=True)
        self.temp_module = TempModule(channels, heads, is_pri=True)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.ReLU(),
            nn.Linear(channels*2, channels)
        ).apply(init_weights_xavier)

        self.norm = nn.Sequential(
                Rearrange('b t n f-> b f (t n)'),
                nn.GroupNorm(4, channels),
                Rearrange('b f (t n) -> b t n f', t=24),
                nn.ReLU()
            ).apply(init_weights_xavier)

    def forward(self, x, edges, weights, side_info, support):

        x = self.initial_conv(x)
        side_info = self.initial_conv_side_info(side_info)

        h = x + side_info
        h = rearrange(h, 'b f t n -> b t n f')

        h_gcn, h_att = self.spa_module(h, edges, weights, support=support)
        h_temp = self.temp_module(h)

        h_final = h_temp + h_gcn + h_att

        h_prior = self.mlp(h_final) + h_final
        return self.norm(h_prior)

class NEM(nn.Module):
    '''Noise Estimation Module from priSTI'''
    def __init__(self, channels, heads, t_emb_size, side_info_dim=144):
        super().__init__()
        self.temp_module = TempModule(channels, heads)
        self.spa_module = SpaModule(channels, heads)

        self.cond_projection = Conv2DCustom(side_info_dim, 2*channels, reorder_out=False)
        self.mid_projection = Conv2DCustom(channels, 2*channels, reorder_out=False)
        self.output_projection = Conv2DCustom(channels, 2*channels, reorder_in=False)
        self.t_emb_compresser = nn.Linear(t_emb_size, channels).apply(init_weights_xavier)

        self.sqrt_two = torch.sqrt(torch.tensor(2.0))

    def forward(self, h_in, h_pri, t_pos, side_info, edges, weights, support):

        t_pos_com = self.t_emb_compresser(t_pos)
        h_in_pos = h_in + t_pos_com

        h_tem = self.temp_module(qk=h_pri, v=h_in_pos)
        h_spa = self.spa_module(h_tem, edges, weights, h_pri, support=support) # (B, T, N, F)
        h = self.mid_projection(h_spa) # (B, 2*F, T, N)

        h_side_info = self.cond_projection(side_info) # (B, 2*F, T, N)
        h += h_side_info # (B, 2*F, T, N)

        # Gates
        gate, filter_gate = torch.chunk(h, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_gate)
        y = self.output_projection(y) # (B, T, N, 2*F)

        residual, skip = torch.chunk(y, 2, dim=-1) # residual: (B, T, N, F), skip: (B, T, N, F)
        
        residual = (h_in + residual)/self.sqrt_two

        return residual, skip
    
class DTigre(nn.Module):
    def __init__(self, channels=64, heads=8, t_emb_size=128):
        super().__init__()

        self.side_info = SideInfo(
            batch_size=4, 
            time_steps=24, 
            num_nodes=207)
        
        self.t_encoder = TEncoder(t_emb_size)
        self.cfem = CFEM(channels, heads)

        self.input_projection = nn.Sequential(
            Conv2DCustom(2, channels),
            nn.ReLU()
        )

        self.nem_layers = nn.ModuleList([
            NEM(channels, heads, t_emb_size) for _ in range(4)
            ])

        self.out = nn.Sequential(
            Conv2DCustom(channels, channels, reorder_out=False),
            nn.ReLU(),
            Conv2DCustom(channels, 1, reorder_in=False)
        )

        self.support = torch.load('support.pt') # Esto debería de eliminarlo
        self.support[0] = self.support[0]
        self.support[1] = self.support[1]
        self.nodevec1 = nn.Parameter(torch.randn(207, 10), requires_grad=True) 
        self.nodevec2 = nn.Parameter(torch.randn(10, 207), requires_grad=True)
        self.support.append([self.nodevec1, self.nodevec2])

    def forward(self, x_ta_t, x_co, u, t, edges, weights):

        side_info = self.side_info(u)

        #mask = cond_info['mask_co']
        h_in = torch.cat([x_ta_t, x_co], dim=-1) # (B, T, N, 2)
        h_in = self.input_projection(h_in)
        
        h_pri = self.cfem(x_co, edges, weights, side_info, self.support)

        t = t.unsqueeze(-1).type(torch.float)
        
        t_pos = self.t_encoder(t)
        t_pos = rearrange(t_pos, 'b f -> b 1 1 f').repeat((1, h_in.shape[1], h_in.shape[2], 1)) # Resample t to (B, T, N, F)

        skip_connections = []
        for nem in self.nem_layers:
            h_in, skip = nem(h_in, h_pri, t_pos, side_info, edges, weights, self.support)
            skip_connections.append(skip)

        res = torch.stack(skip_connections)
        len_layers = torch.tensor(len(self.nem_layers)).to(res.device)
        res = torch.sum(res, dim=0) / torch.sqrt(len_layers)

        return self.out(res)
    