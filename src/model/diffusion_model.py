import torch
import torch.nn as nn
from tsl.nn.models.base_model import BaseModel
from noise_schedulers import LinearNoiseScheduler


class DiffusionModel(BaseModel):
    def __init__(self, noise_steps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.noise_scheduler = LinearNoiseScheduler(noise_steps)
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)

    def apply_noise(self, x, t):
        alpha_hat = self.noise_scheduler.get_alpha_hat(t)
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_hat) * x + torch.sqrt(1 - alpha_hat) * noise
        return noisy_x, noise

    