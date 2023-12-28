import torch

class LinearNoiseScheduler:
    def __init__(self, noise_steps, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps

        self.betas = torch.linspace(beta_start, beta_end, noise_steps)

        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def get_alpha(self, step):
        return self.alphas[step]
    
    def get_beta(self, step):
        return self.betas[step]
    
    def get_alpha_hat(self, step):
        return self.alpha_hat[step]

    