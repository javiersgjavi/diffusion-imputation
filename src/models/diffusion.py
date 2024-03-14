import torch
from tqdm import tqdm
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics
from src.models.tgnn_bi_hardcoded import BiModel
from src.models.pristi import PriSTI

from src.models.data_handlers import RandomStack, SchedulerCSDI, SchedulerPriSTI
    
class DiffusionImputer(Imputer):
    def __init__(self, scheduler_type='squaredcos_cap_v2', *args, **kwargs):
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
        self.model = PriSTI()
        self.scheduler = SchedulerPriSTI(
            num_train_timesteps=self.num_T,
            beta_schedule=scheduler_type
        )
        
    def get_imputation(self, batch):
        x_co_0 = batch.x_interpolated
        u = batch.u
        mask_co = batch.mask
        transform = batch.transform
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        x_ta_t, cond_info, _ = self.scheduler.prepare_data(x_co_0, mask_co, u)
        
        steps_chain = range(0, self.num_T)
        pbar = tqdm(steps_chain, desc=f'[INFO] Imputing batch...')
        for i in reversed(steps_chain):
            t = (torch.ones(x_ta_t.shape[0]) * i).to(x_ta_t.device)
            noise_pred = self.model(x_ta_t, cond_info, t, edge_index, edge_weight)
            x_ta_t = self.scheduler.clean_backwards(x_ta_t, noise_pred, mask_co, t)
            pbar.update(1)
        pbar.close()

        x_0 = transform['x'].inverse_transform(x_ta_t)
        return x_0
    
    def calculate_loss(self, batch):
        x_co_0 = batch.x_interpolated
        x_real_0 = batch.transform['y'](batch.y)
        u = batch.u
        mask_co = batch.mask
        mask_ta = batch.eval_mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        t = self.t_sampler.get(x_co_0.shape[0]).to(x_co_0.device)
        x_ta_t, cond_info, noise = self.scheduler.prepare_data(x_co_0, mask_co, u, t=t, x_real_0=x_real_0)

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
        if batch_idx > 10:
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
