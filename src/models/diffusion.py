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
        self.num_T = 50
        self.t_sampler = RandomStack(1, self.num_T, device=self.device)
        self.loss_fn = torch_metrics.MaskedMSE()
        self.model = BiModel()#UNet
        self.scheduler = Scheduler(
            beta_schedule=scheduler_type,
            num_train_timesteps=self.num_T
            )

    def log_metrics(self, metrics, **kwargs):
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      logger=True,
                      prog_bar=False,
                      **kwargs)

    def log_loss(self, name, loss, **kwargs):
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=True,
                 on_epoch=True,
                 logger=True,
                 prog_bar=True,
                 **kwargs)
        

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]


    def calculate_loss(self, noise, noise_pred, noise_f_pred, noise_b_pred, eval_mask):
        loss_p = self.loss_fn(noise, noise_pred, eval_mask)
        custom_loss = False
        
        if custom_loss:
            loss_f_0 = self.loss_fn(noise, noise_f_pred[0], eval_mask)
            loss_f_1 = self.loss_fn(noise, noise_f_pred[1], eval_mask)

            loss_b_0 = self.loss_fn(noise, noise_b_pred[0], eval_mask)
            loss_b_1 = self.loss_fn(noise, noise_b_pred[1], eval_mask)

            loss = loss_p + loss_f_0 + loss_f_1 + loss_b_0 + loss_b_1
            loss = torch.mean(loss)
        else:
            loss = loss_p
        return loss, loss_p
    
    def obtain_data_masked(self, x_co_0, x_real_0, mask_co, u, t):
        # Esto se supone que es la Figura 5 del paper de CSDI
        zero = torch.zeros_like(x_co_0)

        x_noisy_t, noise = self.scheduler.forward(x_real_0, t) # esto puede ser un punto de fallo
        x_ta_t = torch.where(mask_co.bool(), zero, x_noisy_t)

        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, x_co_0.shape[2], 1)
        cond_info = torch.cat([x_co_0, mask_co, u], dim=-1)

        return x_ta_t, cond_info, noise

        
    def get_imputation(self, batch):
        
        # más o menos, pero va a haber que actualizarlo (Se podría pasar a veces None como condicional y ver si funciona como CFG)
        x_co_0 = batch.x
        # x_real_0 = batch.transform['y'](batch.y)
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask_co = batch.mask
        # mask_ta = batch.eval_mask
        transform = batch.transform

        zero = torch.zeros_like(x_co_0)
        noise = torch.randn_like(x_co_0)

        x_ta_t = torch.where(mask_co.bool(), zero, noise)

        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, x_co_0.shape[2], 1)
        cond_info = torch.cat([x_co_0, mask_co, u], dim=-1)

        steps_chain = range(0, self.num_T)
        pbar = tqdm(steps_chain, desc=f'[INFO] Imputing batch...')
        for i in reversed(steps_chain):
            t = (torch.ones(x_ta_t.shape[0]) * i)
            noise_pred = self.model(x_ta_t, x_co_0, t, cond_info, edge_index, edge_weight)[0]
            x_ta_t = self.scheduler.backwards(x_ta_t, noise_pred, t)
            pbar.update(1)

        x_0 = transform['x'].inverse_transform(x_ta_t)
        return x_0, x_ta_t
    
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
        x_ta_t, cond_info, noise = self.obtain_data_masked(x_co_0, x_real_0, mask_co, u, t)

        noise_pred, noise_f_pred, noise_b_pred = self.model(x_ta_t, x_co_0, t, cond_info, edge_index, edge_weight)

        loss, loss_p = self.calculate_loss(noise, noise_pred, noise_f_pred, noise_b_pred, mask_ta)

        # Update metrics
        #self.train_metrics.update(noise, noise_pred, mask_ta) En verdad si estoy mirando el ruido no me interesa tanto saber el mae ni nada de eso
        #self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        self.log_loss('train_p', loss_p, batch_size=batch.batch_size)
        return loss

    '''def validation_step(self, batch, batch_idx):
        x_t = self.get_imputation(batch)
        target = batch.y
        eval_mask = batch.eval_mask
    
        loss = self.loss_fn(x_t, target, eval_mask)

        # Update metrics
        self.val_metrics.update(x_t, target, eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)'''
    
    def validation_step(self, batch, batch_idx):
        x_co_0 = batch.x
        x_real_0 = batch.transform['y'](batch.y)
        u = batch.u
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        mask_co = batch.mask
        mask_ta = batch.eval_mask


        t = self.t_sampler.get(x_co_0.shape[0]).to(x_co_0.device)
        x_ta_t, cond_info, noise = self.obtain_data_masked(x_co_0, x_real_0, mask_co, u, t)

        noise_pred, noise_f_pred, noise_b_pred = self.model(x_ta_t, x_co_0, t, cond_info, edge_index, edge_weight)

        loss, loss_p = self.calculate_loss(noise, noise_pred, noise_f_pred, noise_b_pred, mask_ta)

        # Update metrics
        #self.val_metrics.update(noise, noise_pred, mask_ta)
        #self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):
        x_t = self.get_imputation(batch)
        target = batch.y
        eval_mask = batch.eval_mask

        # Update metrics
        self.test_metrics.update(x_t, target, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)