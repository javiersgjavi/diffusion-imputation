import torch
from tqdm import tqdm
from diffusers import DDPMScheduler
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics
from src.models.tgnn_bi_hardcoded import BiModel
from src.models.pristi import PriSTI

class RandomStack:
    def __init__(self, high, low=0, size=1e6):
        self.size = int(size)
        self.high = high
        self.low = low
        self.stack = torch.randint(low=low, high=high, size=(self.size,))
        self.idx = 0

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
        step = step[0].int()
        return super().step(predicted_noise, step, x_t).prev_sample
    
class DiffusionImputer(Imputer):
    def __init__(self, scheduler_type='squaredcos_cap_v2', interpolated=False, *args, **kwargs):
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
        super().__init__(*args, **kwargs)
        self.num_T = 50
        self.masked_mae = torch_metrics.MaskedMAE()
        self.t_sampler = RandomStack(self.num_T)
        self.loss_fn = torch_metrics.MaskedMSE()
        self.interpolated = interpolated
        self.model = PriSTI()
        self.scheduler = Scheduler(
            num_train_timesteps=self.num_T,
            beta_schedule=scheduler_type
        )

    def obtain_data_masked(self, x_co_0, mask_co, u, t=None, x_real_0=None):

        if x_real_0 is None:
            noise = torch.randn_like(x_co_0)
            x_noisy_t = noise
        else:
            x_noisy_t, noise = self.scheduler.forward(x_real_0, t)

        zeros = torch.zeros_like(x_co_0)
        x_ta_t = torch.where(mask_co.bool(), zeros, x_noisy_t)
    
        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, x_co_0.shape[2], 1)
        cond_info = {
            'x_co': x_co_0,
            'mask_co': mask_co,
            'u': u,
        }

        return x_ta_t, cond_info, noise
        
    def get_imputation(self, batch):
        x_co_0 = batch.x
        u = batch.u
        mask_co = batch.mask
        transform = batch.transform
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        zeros = torch.zeros_like(x_co_0)

        x_ta_t, cond_info, _ = self.obtain_data_masked(x_co_0, mask_co, u)
        
        steps_chain = range(0, self.num_T)
        pbar = tqdm(steps_chain, desc=f'[INFO] Imputing batch...')
        for i in reversed(steps_chain):
            t = (torch.ones(x_ta_t.shape[0]) * i).to(x_ta_t.device)
            noise_pred = self.model(x_ta_t, cond_info, t, edge_index, edge_weight)
            x_ta_t = self.scheduler.backwards(x_ta_t, noise_pred, t)
            x_ta_t = torch.where(mask_co.bool(), zeros, x_ta_t)
            pbar.update(1)
        pbar.close()

        x_0 = transform['x'].inverse_transform(x_ta_t)
        return x_0
    
    def calculate_loss(self, batch):
        x_co_0 = batch.x
        x_real_0 = batch.transform['y'](batch.y)
        u = batch.u
        mask_co = batch.mask
        mask_ta = batch.eval_mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        t = self.t_sampler.get(x_co_0.shape[0]).to(x_co_0.device)
        x_ta_t, cond_info, noise = self.obtain_data_masked(x_co_0, mask_co, u, t=t, x_real_0=x_real_0)

        noise_pred  = self.model(x_ta_t, cond_info, t, edge_index, edge_weight)
        return self.loss_fn(noise, noise_pred, mask_ta)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log_loss('val', loss, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):
        if batch_idx > 2:
            return torch.tensor(0.0)
        x_t = self.get_imputation(batch)
        self.test_metrics.update(x_t, batch.y, batch.eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return [optim], [scheduler]
    
    def log_metrics(self, metrics, **kwargs):
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        self.log(
            name + '_loss',
            loss.detach(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            **kwargs
        )