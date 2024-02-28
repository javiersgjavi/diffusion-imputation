import torch
from torch import nn

from tsl.nn.blocks.decoders import GCNDecoder

from src.utils import init_weights_xavier, clean_hyperparams, get_encoder, define_mlp_decoder


class TEncoder(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

    def forward(self, x, edges, weights):
        pass

class ConditionalEncoder(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

    def forward(self, x, edges, weights, exog):
        pass

class UniModel(nn.Module):
    def __init__(self, hyperparameters, d = False):
        super().__init__()
        self.name = hyperparameters['encoder_name']

        self.encoder = get_encoder[hyperparameters['encoder_name']](**hyperparameters['encoder'])
        self.decoder = GCNDecoder(**hyperparameters['decoder'])

        self.encoder.apply(init_weights_xavier)
        self.decoder.apply(init_weights_xavier)

    def forward(self, x, edges, weights):

        x = self.encoder(x) if self.name != 'stcn' else self.encoder(x, edges, weights)
        x = self.decoder(x, edges, weights)
        return x

class BiModel(nn.Module):
    '''def __init__(self, args):
        super().__init__(edge_index=args['edge_index'], edge_weights=args['edge_weights'])

        self.args = args
        print(self.args)
        self.output_size_decoder = int(args['periods'] * args['generator']['mlp']['hidden_size'])//2
        
        self.args = clean_hyperparams(self.args)

        self.model_f = UniModel(self.args)
        self.model_b = UniModel(self.args)

        define_mlp_decoder(self.args['generator']['mlp'])

        print(self.model_f)
        print(self.decoder_mlp)'''
    
    def __init__(self, args):
        args={

        }
        self.model_f = UniModel(args)
   
    def forward(self,x, t, cond_info, edges, weights):
        
        cond_emb_f = self.cond_encoder_f(t, cond_info)
        cond_emb_b = self.cond_encoder_b(t, cond_info)

        f_representation = self.model_f(x, cond_emb_f)
        b_representation = self.model_b(x, cond_emb_b)

        h = torch.cat([f_representation, b_representation, cond_emb_f], dim=-1)
        output = self.decoder_mlp(h)
        return output
    