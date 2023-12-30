import torch

class LinearNoiseScheduler:
    def __init__(self, noise_steps, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.noise_steps = noise_steps

        self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(device)

        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0).to(device)

    def get_alpha(self, steps, like=None):
        return self.__get_data(self.alphas, steps, like)
    
    def get_beta(self, steps, like=None):
        return self.__get_data(self.betas, steps, like)
    
    def get_alpha_hat(self, steps, like=None):
        return self.__get_data(self.alpha_hat, steps, like)

    def __get_data(self, tensor, steps, like=None):
        shape=like.shape
        vector = tensor[steps].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        res = vector.expand(-1, shape[1], shape[2], -1)
        return res