import torch
import torch.nn as nn
from tsl.nn.models.base_model import BaseModel
from noise_schedulers import LinearNoiseScheduler


class DiffusionModel(BaseModel):
    def __init__(self, noise_steps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.noise_steps = noise_steps
        self.noise_scheduler = LinearNoiseScheduler(noise_steps)
        self.model = None

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)

    def apply_noise(self, x, t):
        alpha_hat = self.noise_scheduler.get_alpha_hat(t)
        ε = torch.randn_like(x)
        x_t = torch.sqrt(alpha_hat) * x + torch.sqrt(1 - alpha_hat) * ε
        return x_t, ε
    
    def predict_noise(self, batch, t):
        x_co = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask
        
        x_t, ε = self.apply_noise(x_co, t)

        ε_pred = self.model.forward(x_t, mask, edge_index, edge_weight, t, u, x_co)
        
        return ε, ε_pred
    
    def sample(self, batch):
        x_co = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask

        x = torch.randn_like(x_co)

        for t in reversed(range(1, self.noise_steps)):
            ε_pred = self.model.forward(x, mask, edge_index, edge_weight, t, u, x_co)
            α_t = self.noise_scheduler.get_alpha(t)
            α_hat_t = self.noise_scheduler.get_alpha_hat(t)
            β = self.noise_scheduler.get_beta(t)

            if t > 1:
                ε = torch.randn_like(x)
            else:
                ε = torch.zeros_like(x)

            x = 1/torch.sqrt(α_t) * (x - ((1 - α_t)/(torch.sqrt(1 - α_hat_t))) * ε_pred) + torch.sqrt(β)*ε


        return x

