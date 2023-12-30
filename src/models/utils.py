from torch import nn

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