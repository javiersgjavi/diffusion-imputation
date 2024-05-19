import torch
from omegaconf import open_dict
from torch import Tensor


from contextlib import nullcontext
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics

from src.models.csdi import diff_CSDI
from models.pristi import PriSTI
from src.models.timba import Timba

from src.data.data_handlers import RandomStack, SchedulerPriSTI, MissingPatternHandler, create_interpolation, redefine_eval_mask

from schedulefree import AdamWScheduleFree
from torch_ema import ExponentialMovingAverage

from src.utils import print_summary_model

class DiffusionImputer(Imputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.masked_mae = torch_metrics.MaskedMAE()
        self.loss_fn = torch_metrics.MaskedMSE()

        scheduler_kwargs = kwargs['model_kwargs'].pop('scheduler_kwargs')
        self.num_T = scheduler_kwargs['num_train_timesteps']
        
        self.t_sampler = RandomStack(self.num_T, dtype_int=True)
        self.scheduler = SchedulerPriSTI(**scheduler_kwargs)

        model_hyperparams = self.model_kwargs.pop('config')
        with open_dict(model_hyperparams):
            model_hyperparams.num_steps = self.num_T

        self.model = diff_CSDI(config = model_hyperparams)

        self.use_ema = self.model_kwargs['use_ema']
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.model_kwargs['decay']) if self.use_ema else None

        self.missing_pattern_handler = MissingPatternHandler(
            strategy1=self.model_kwargs['missing_pattern']['strategy1'], 
            strategy2=self.model_kwargs['missing_pattern']['strategy2'], 
            hist_patterns=self.model_kwargs['hist_patterns'],
            seq_len=model_hyperparams['time_steps']
            )

        # print_summary_model(self.model, model_hyperparams)
        
    def get_imputation(self, batch):
        mask_co = batch.mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        x_ta_t, cond_info, _ = self.scheduler.prepare_data(batch)
        
        for i in reversed(range(self.num_T)):
            t = (torch.ones(x_ta_t.shape[0]) * i).to(x_ta_t.device)
            noise_pred = self.model(x_ta_t, cond_info['x_co'], cond_info['mask_co'], t)
            x_ta_t = self.scheduler.clean_backwards(x_ta_t, noise_pred, mask_co, t)

        x_0 = batch.transform['x'].inverse_transform(x_ta_t)
        return x_0
    
    def calculate_loss(self, batch, t=None):
        mask_ta = batch.eval_mask
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight

        t = self.t_sampler.get(mask_ta.shape[0]).to(mask_ta.device) if t is None else t
        x_ta_t, cond_info, noise = self.scheduler.prepare_data(batch,t=t)

        noise_pred  = self.model(x_ta_t, cond_info['x_co'], cond_info['mask_co'], t)

        return self.loss_fn(noise, noise_pred, mask_ta)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters() if self.use_ema else nullcontext():
            loss = torch.zeros(1).to(batch.x.device)
            for t in range(self.num_T):
                t = (torch.ones(batch.x.shape[0]) * t).to(batch.x.device)
                loss += self.calculate_loss(batch, t)

        loss /= self.num_T
        self.log_loss('val', loss, batch_size=batch.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        x_t_list = []

        for _ in range(100):
            x_t = self.get_imputation(batch)
            x_t_list.append(x_t)

        x_t = torch.cat(x_t_list, dim=-1)
        x_t = x_t.median(dim=-1).values.unsqueeze(-1)
        
        self.test_metrics.update(x_t, batch.y, batch.eval_mask)
        print(self.masked_mae(x_t, batch.y, batch.eval_mask))
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
    
    def log_metrics(self, metrics, **kwargs):
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        self.log(
            name + '_loss',
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            **kwargs
        )

    def on_validation_batch_start(self, batch, batch_idx: int) -> None:
        super().on_validation_batch_start(batch, batch_idx)
        batch = create_interpolation(batch)

    def on_test_batch_start(self, batch, batch_idx: int) -> None:
        super().on_test_batch_start(batch, batch_idx)
        batch = create_interpolation(batch)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().train()

        if self.use_ema:
            if self.ema.shadow_params[0].device != self.device:
                self.ema.to(self.device)

    def on_train_batch_end(self, *args, **kwargs)-> None:
        super().on_train_batch_end(*args, **kwargs)
        if self.use_ema:
            self.ema.update()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().eval()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        if self.optim_class == AdamWScheduleFree:
            self.optimizers().eval()

    def parameters(self):
        return self.model.parameters()
    
    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx)
        self.missing_pattern_handler.update_mask(batch)

        batch = create_interpolation(batch)
        batch = redefine_eval_mask(batch)
