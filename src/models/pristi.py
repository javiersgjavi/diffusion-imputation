from src.models.layers_pristi import *
from src.utils_pristi import *

from src.models.layers import CustomMamba

class SideInfo(nn.Module):
    def __init__(self, time_steps, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.embed_layer = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=16)

        self.arange = torch.arange(self.num_nodes)
    
    def get_time(self, B):
        observed_tp = torch.arange(self.time_steps).unsqueeze(0)
        pos = torch.cat([observed_tp for _ in range(B)], dim=0)
        self.div_term = 1 / torch.pow(
            10000.0, torch.arange(0, 128, 2) / 128
        )
        pe = torch.zeros(pos.shape[0], pos.shape[1], 128)
        position = pos.unsqueeze(2)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, cond_mask):
        B, _, K, L = cond_mask.shape

        observed_tp= torch.arange(L).to(cond_mask.device).unsqueeze(0)
        observed_tp = torch.cat([observed_tp for _ in range(B)], dim=0)
        time_embed = self.get_time(B).unsqueeze(2).expand(-1, -1, K, -1).to(cond_mask.device)
        self.arange = self.arange.to(cond_mask.device)
        feature_embed = self.embed_layer(self.arange)  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info.to(cond_mask.device)

class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True, num_nodes=None, time_steps=None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, itp_info)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

class PriSTI(nn.Module):
    def __init__(self, inputdim=2, is_itp=True, config=None):
        super().__init__()
        
        self.num_nodes = config['num_nodes']
        self.channels = config["channels"]
        self.time_steps = config["time_steps"]
        self.batch_size = config["batch_size"]

        self.side_info = SideInfo(self.time_steps, self.num_nodes)

        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=config["nheads"], target_dim=self.num_nodes,
                                            order=2, include_self=True, device=None, is_adp=config["is_adp"],
                                            adj_file=config["adj_file"], proj_t=config["proj_t"], time_steps = config["time_steps"], num_nodes = config['num_nodes'])
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        elif config["adj_file"] == 'mimic-iii':
            self.adj = get_similarity_mimic(thr=0.1)

        self.support = compute_support_gwn(self.adj)
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=self.num_nodes,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=None,
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                    time_steps = config["time_steps"],
                    num_nodes = config['num_nodes']
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, itp_x, u, diffusion_step):


        if self.is_itp:
            x = torch.cat([x, itp_x], dim=-1)

        x = x.permute(0, 3, 2, 1) # B, input_dim, K, 
        side_info = self.side_info(x)

        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1, K*L)
        x = x.reshape(B, 1, K, L).permute(0, 3, 2, 1)

        return x