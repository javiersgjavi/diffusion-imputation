import torch
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics

from src.models.diffusion_model import DiffusionModel

class TimeStepSampler:
    def __init__(self, noise_step=1000, size=1e5, device='cpu'):
        self.noise_step = noise_step
        self.size = int(size)
        self.device = device
        self.timesteps = torch.randint(low=1, high=self.noise_step, size=(self.size,)).to(self.device)
        self.i = 0

    def __check_t(self, n):
        if self.i + n >= len(self.timesteps):
            self.timesteps = torch.randint(low=0, high=self.noise_step, size=(self.size,)).to(self.device)
            self.i = 0

    def sample(self, n):
        self.__check_t(n)
        value = self.timesteps[self.i:self.i+n]
        self.i += n
        return value
    
class DiffusionImputer(Imputer):
    def __init__(self, *args, **kwargs):
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            #'rmse': torch_metrics.MaskedRMSE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
        super().__init__(*args, **kwargs)
        self.t_sampler = TimeStepSampler(1000, device=self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.model = DiffusionModel(device=self.device)

    def training_step(self, batch, batch_idx):  
        #y = y_loss = batch.y
        eval_mask = batch.get('eval_mask')
        
        t = self.t_sampler.sample(eval_mask.shape[0])

        # Compute predictions and compute loss
        ε, ε_pred = self.model.predict_noise(batch, t)

        #print('Prediction', round(torch.mean(ε_pred[eval_mask]).item(), 2), round(torch.median(ε_pred[eval_mask]).item(), 2), round(torch.std(ε_pred[eval_mask]).item(), 2))

        #print(torch.stack([ε[eval_mask], ε_pred[eval_mask]], dim=-1))
        # Compute loss
        loss = self.loss_fn(ε[eval_mask], ε_pred[eval_mask])

        #print(f'Loss: {loss.item()}')

        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #y = y_loss = batch.y
        mask = batch.get('mask')
        eval_mask = batch.get('eval_mask')
        
        t = self.t_sampler.sample(eval_mask.shape[0])

        # Compute predictions and compute loss
        ε, ε_pred = self.model.predict_noise(batch, t)

        # Compute loss
        loss = self.loss_fn(ε[eval_mask], ε_pred[eval_mask])

        # Logging
        #self.val_metrics.update(y_hat, y, mask)
        #self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        #y = y_loss = batch.y
        x = batch.x
        mask = batch.get('mask')
        eval_mask = batch.get('eval_mask')

        # Compute predictions and compute loss
        imputation = self.model.sample(batch)

        x_imputated = batch.x * mask + imputation * (1 - mask)

        # Compute loss
        loss = torch.zeros(1)

        # Logging
        self.test_metrics.update(x_imputated, x, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', loss, batch_size=batch.batch_size)
        return loss
