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
        self.loss_fn = torch_metrics.MaskedMSE()
        self.model = None
        self.scheduler = Scheduler()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]
    
    def training_step(self, batch, batch_id):
        # más o menos, pero va a haber que actualizarlo (Se podría pasar a veces None como condicional y ver si funciona como CFG)
        x = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask
        eval_mask = batch.eval_mask

        t = self.t_sampler.get(x.shape[0])

        x_t, noise = self.scheduler.forward()

        x_t = torch.where(mask.bool(), x, x_t)

        noise_pred = self.model(x_t, t)#(x_t, t, mask, edge_index, edge_weight)

        loss = self.loss_fn(noise, noise_pred, eval_mask)

        self.log('train_loss', loss, prog_bar=True, on_step=True)#, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        return loss
    
    def impute(self, batch):
        x = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask
        eval_mask = batch.eval_mask

        x_t = torch.where(mask.bool(), x, torch.rand_like(x).to(x.device))

        steps_chain = range(1, self.noise_steps)
        pbar = tqdm(steps_chain, desc=f'[INFO] Imputing batch...')
        for i in reversed(steps_chain):
            t = (torch.ones(x.shape[0]) * i).to(x.device)
            noise_pred = self.model(x_t, t)#(x_t, t, mask, edge_index, edge_weight)
            x_t = self.scheduler.backwards(x_t, noise_pred, t)
            x_t = torch.where(mask.bool(), x, x_t)
            pbar.update(1)

        loss = torch_metrics.MaskedMAE(x, x_t, eval_mask)

        return loss