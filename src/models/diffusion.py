import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics


class RandomStack:
    def __init__(self, high, low, device='cpu', size=1e6):
        self.size = int(size)
        self.high = high
        self.low = low
        self.stack = torch.randint(low=low, high=high, size=(self.size,)).to(device)
        self.idx = 0

    def get(self, n=1):
        if self.idx + n >= self.size:
            self.stack = torch.randint(low=self.low, high=self.high, size=(self.size,)).to(self.stack.device)
            self.idx = 0
        res = self.stack[self.idx: self.idx + n]
        self.idx += n
        return res
    
class Scheduler(DDPMScheduler):

    def forward(self, x, step):
        noise = torch.randn_like(x)
        return super().add_noise(x, noise, step), noise
    
    def backwards(self, x_t, predicted_noise, step):
        step = step[0]
        return super().step(predicted_noise, step, x_t).prev_sample
    

class DiffusionImputer(Imputer):
    def __init__(self, *args, **kwargs):
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            #'rmse': torch_metrics.MaskedRMSE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
        super().__init__(*args, **kwargs)
        self.num_steps = 1000
        self.t_sampler = RandomStack(1, self.num_steps, device=self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.model = None
        self.scheduler = Scheduler()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]
    
    def training_step(self, batch, batch_id):
        # más o menos, pero va a haber que actualizarlo
        x, y = batch
        t = self.t_sampler.get(x.shape[0])
        x_t, noise = self.scheduler.forward(x, t)
        noise_pred = self.model(x_t, t)
        loss = self.loss_fn(noise, noise_pred)
        self.log('train_loss', loss, prog_bar=True, on_step=True)#, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        return loss
    
    def impute(self, batch):
        x, y = batch
        mask = None
        pbar = tqdm(range(1, self.noise_steps), desc=f'[INFO] Imputing batch...')
        x_t = torch.randn_like(x).to(x.device)
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(x.shape[0]) * i).to(x.device)
            noise_pred = self.model(x_t, t)
            x_t = self.scheduler.backwards(x_t, noise_pred, t)
            pbar.update(1)
        return torch_metrics.MaskedMAE()(x, x_t, mask)
