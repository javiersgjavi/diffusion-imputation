import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics
from src.models.unet import UNet
from src.models.tgnn_bi_hardcoded import BiModel

class RandomStack:
    def __init__(self, low, high, device='cpu', size=1e6):
        self.size = int(size)
        self.high = high
        self.low = low
        self.stack = torch.randint(low=low, high=high, size=(self.size,))
        self.idx = 0
        self.device = device

    def get(self, n=1):
        if self.idx + n >= self.size:
            self.stack = torch.randint(low=self.low, high=self.high, size=(self.size,))
            self.idx = 0
        res = self.stack[self.idx: self.idx + n]
        self.idx += n
        return res
    
class Scheduler(DDPMScheduler):
    def forward(self, x, step):
        noise = torch.randn_like(x)
        return super().add_noise(x, noise, step), noise
    
    def backwards(self, x_t, predicted_noise, step):
        # mirar si puedo quitarme el .cpu()
        step = step[0].int()
        device = self.alphas_cumprod.device
        return super().step(predicted_noise.to(device), step.to(device), x_t.to(device)).prev_sample.to(x_t.device)
    

class DiffusionImputer(Imputer):
    def __init__(self, scheduler_type='squaredcos_cap_v2', *args, **kwargs):
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            #'rmse': torch_metrics.MaskedRMSE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
        super().__init__(*args, **kwargs)
        self.num_T = 1000
        self.t_sampler = RandomStack(1, self.num_T, device=self.device)
        self.loss_fn = torch_metrics.MaskedMSE()
        self.model = BiModel()#UNet
        self.scheduler = Scheduler(beta_schedule=scheduler_type)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]


    def obtain_data_masked(self, x_co_0, x_real_0, mask_co, mask_ta, t):
        # Esto se supone que es la Figura 5 del paper de CSDI
        zero = torch.zeros_like(x_co_0)

        x_0_ta = torch.where(mask_ta, x_real_0, zero)

        x_noisy_t, noise = self.scheduler.forward(x_real_0, t)
        x_ta_t = torch.where(mask_co.bool(), zero, x_noisy_t)

        cond_info = torch.cat([x_0_ta, mask_co], dim=-1)
        return x_ta_t, cond_info, noise

    def training_step(self, batch, batch_id):
        # más o menos, pero va a haber que actualizarlo (Se podría pasar a veces None como condicional y ver si funciona como CFG)
        x_co_0 = batch.x
        x_real_0 = batch.transform['y'](batch.y)
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask_co = batch.mask
        mask_ta = batch.eval_mask


        t = self.t_sampler.get(x_co_0.shape[0]).to(x_co_0.device)
        x_ta_t, cond_info, noise = self.obtain_data_masked(x_co_0, x_real_0, mask_co, mask_ta, t)

        noise_pred = self.model(x_ta_t, t, cond_info, edge_index, edge_weight)

        loss = self.loss_fn(noise, noise_pred, mask_ta)

        self.log('train_loss', loss, prog_bar=True, on_step=True)#, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        '''x_t = self.get_imputation(batch)
        self.val_metrics(x_t, batch.x, batch.eval_mask)'''

    def test_step(self, batch, batch_idx):
        x_t = self.get_imputation(batch)
        self.test_metrics(x_t, batch.x, batch.eval_mask)
    
    def get_imputation(self, batch):
        
        x = batch.x
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask = batch.mask
        eval_mask = batch.eval_mask
        cond_info = torch.cat([x, mask], dim=-1)

        x_t = torch.where(mask.bool(), x, torch.rand_like(x).to(x.device))

        steps_chain = range(1, self.num_T//100)
        #pbar = tqdm(steps_chain, desc=f'[INFO] Imputing batch...')
        for i in reversed(steps_chain):
            t = (torch.ones(x.shape[0]) * i)
            noise_pred = self.model(x_t, t, cond_info, edge_index, edge_weight)
            x_t = self.scheduler.backwards(x_t, noise_pred, t)
            x_t = torch.where(mask.bool(), x, x_t)
            #pbar.update(1)

        return x_t