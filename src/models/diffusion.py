import torch
from tqdm import tqdm
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics
from torchinfo import summary

from src.models.tgnn_bi_hardcoded import BiModel
from src.models.pristi import PriSTI
from src.models.pristi_o import PriSTIO

from src.models.data_handlers import RandomStack, SchedulerPriSTI, create_interpolation, redefine_eval_mask
    
class DiffusionImputer(Imputer):
    def __init__(self, scheduler_type='scaled_linear', *args, **kwargs):
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
            beta_schedule=scheduler_type,
            beta_start=0.0001,
            beta_end=0.2,
            clip_sample=False,
        )
        
        summary(
            self.model,
            input_size=[(4, 24, 207, 1), (4, 24, 207, 1), (4, 24, 207, 2), (4,), (2, 1515), (1515,)],
            dtypes=[torch.float32, torch.float32, torch.float32, torch.int64, torch.int64, torch.float32],
            col_names=['input_size', 'output_size', 'num_params'],
            depth=2
            )
        
    def get_imputation(self, batch):
        mask_co = batch.mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        x_ta_t, cond_info, _ = self.scheduler.prepare_data(batch)
        
        for i in reversed(range(self.num_T)):
            t = (torch.ones(x_ta_t.shape[0]) * i).to(x_ta_t.device)
            noise_pred = self.model(x_ta_t, cond_info['x_co'], cond_info['u'], t, edge_index, edge_weight)
            x_ta_t = self.scheduler.clean_backwards(x_ta_t, noise_pred, mask_co, t)

        x_0 = batch.transform['x'].inverse_transform(x_ta_t)
        return x_0
    
    def calculate_loss(self, batch, t=None):
        mask_ta = batch.eval_mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        t = self.t_sampler.get(mask_ta.shape[0]).to(mask_ta.device) if t is None else t
        x_ta_t, cond_info, noise = self.scheduler.prepare_data(batch,t=t)

        noise_pred  = self.model(x_ta_t, cond_info['x_co'], cond_info['u'], t, edge_index, edge_weight)

        residual = (noise - noise_pred) * mask_ta
        num_eval = mask_ta.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def training_step(self, batch, batch_idx):
        batch = create_interpolation(batch)
        batch = redefine_eval_mask(batch)

        loss = self.calculate_loss(batch)
        self.log_loss('train', loss, batch_size=batch.batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = create_interpolation(batch)

        loss = torch.zeros(1).to(batch.x.device)
        for t in range(self.num_T):
            t = (torch.ones(batch.x.shape[0]) * t).to(batch.x.device)
            loss += self.calculate_loss(batch, t)

        loss /= self.num_T
        self.log_loss('val', loss, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):
        batch = create_interpolation(batch)
        x_t_list = []

        for _ in range(100):
            x_t = self.get_imputation(batch)
            x_t_list.append(x_t)

        x_t = torch.cat(x_t_list, dim=-1)
        x_t = x_t.median(dim=-1).values.unsqueeze(-1)
        
        self.test_metrics.update(x_t, batch.y, batch.eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        p1 = int(0.75 * 50)
        p2 = int(0.9 * 50)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[p1, p2], gamma=0.1)
        return [optim], [scheduler]
        #return optim
    
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


