from torch import nn
import torch
import math
from torchinfo import summary

from tsl.nn.blocks.encoders import TemporalConvNet, SpatioTemporalConvNet, Transformer, SpatioTemporalTransformerLayer
from tsl.nn.blocks.encoders.recurrent import RNN, GraphConvRNN#, MultiRNN

def get_activation(name):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'selu': nn.SELU,
        'leaky_relu': nn.LeakyReLU,
    }
    return activations[name]

def get_encoder(name):
    encoders = {
        'rnn': RNN,
        #'mrnn': MultiRNN,
        'tcn': TemporalConvNet, 
        'stcn': SpatioTemporalConvNet, #falla
        'transformer': Transformer,
        'stransformer': SpatioTemporalTransformerLayer, #falla
        'gcrnn': GraphConvRNN 
    }
    return encoders[name]

def round_to_nearest_divisible(x, y):
        """
        This function rounds x to the nearest number divisible by y    
        """
        return round(x / y) * y

def init_weights_xavier(m: nn.Module) -> None:
    """
    Initialize the weights of the neural network module using the Xavier initialization method.

    Args:
        m (nn.Module): Neural network module

    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.01)

def init_weights_kaiming(m: nn.Module) -> None:
    """
    Initialize the weights of the neural network module using the Kaiming initialization method.

    Args:
        m (nn.Module): Neural network module

    Returns:
        None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.01)

def clean_hyperparams(args, output_size_decoder):
    in_features = 2
    encoder_name = args['encoder_name']
    
    args['decoder']['hidden_size'] = int(args['periods'] * args['decoder']['hidden_size'])
    args['decoder']['output_size'] = output_size_decoder
    args['decoder']['horizon'] = args['periods']

    args['mlp']['input_size'] = output_size_decoder*2

    if encoder_name == 'rnn':
        args['encoder']['input_size'] = in_features
        args['encoder']['hidden_size'] = int(args['periods'] * args['encoder']['hidden_size'])
        args['encoder']['output_size'] = int(args['periods'] * args['encoder']['output_size'])
        args['encoder']['exog_size'] = 0

        args['decoder']['input_size'] = args['encoder']['output_size']

    elif encoder_name == 'tcn':
        args['encoder']['input_channels'] = in_features
        args['encoder']['hidden_channels'] = int(args['periods'] * args['encoder']['hidden_channels'])
        args['encoder']['output_channels'] = int(args['periods'] * args['encoder']['output_channels'])

        args['decoder']['input_size'] = args['encoder']['output_channels']

    elif encoder_name == 'stcn':
        args['encoder']['input_size'] = in_features
        args['encoder']['output_size'] = int(args['periods'] * args['encoder']['output_size'])

        args['decoder']['input_size'] = args['encoder']['output_size']

    elif encoder_name == 'transformer':
        args['encoder']['input_size'] = in_features
        hidden_size = int(args['periods'] * args['encoder']['hidden_size'])
        args['encoder']['hidden_size'] = round_to_nearest_divisible(hidden_size, args['encoder']['n_heads'])

        args['encoder']['ff_size'] = int(args['periods'] * args['encoder']['ff_size'])
        args['encoder']['output_size'] = int(args['periods'] * args['encoder']['output_size'])

        args['decoder']['input_size'] = args['encoder']['output_size']

    print('------- FINAL ARGS -------')
    for k, v in args.items():
        print(f'{k}: {v}')

    return args

def define_mlp_decoder(mlp_params):
    decoder_mlp = nn.Sequential()
    mlp_layers = mlp_params['n_layers']
    input_size = mlp_params['input_size']
    activation_fnc = get_activation[mlp_params['activation']]
    ouput_size_final = 24

    for i, l in enumerate(range(mlp_layers, 1, -1)):
        output_size = int(((l - 1) *  (ouput_size_final)/ mlp_layers) + 1)
        output_size = output_size if output_size > 0 else 1
        decoder_mlp.add_module(
            f'linear_{i}',
            nn.Linear(input_size, output_size)
            )
        decoder_mlp.add_module(
            f'activation_{i}',
            activation_fnc()
        )
        decoder_mlp.add_module(
            f'dropout_{i}',
            nn.Dropout(mlp_params['dropout'])
        )
        input_size = output_size

    decoder_mlp.add_module(f'final_linear', nn.Linear(input_size, 1))

    decoder_mlp.apply(init_weights_xavier)

def _init_weights_mamba(
    module,
    n_layer=1,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def print_summary_model(model, params, depth=2):
    print(params)
    t = params.time_steps
    n = params.num_nodes
    b = params.batch_size
    summary(
            model,
            input_size=[(b, t, n, 1), (b, t, n, 1), (b, t, n, 2), (b,), (2, 1515), (1515,)],
            dtypes=[torch.float32, torch.float32, torch.float32, torch.int64, torch.int64, torch.float32],
            col_names=['input_size', 'output_size', 'num_params'],
            depth=depth
            )
