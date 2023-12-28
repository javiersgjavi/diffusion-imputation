import torch
from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics

class TimeStepSampler:
    def __init__(self, noise_step=1000, size=1e5):
        self.noise_step = noise_step
        self.size = size
        self.timesteps = torch.randint(low=0, high=self.noise_step, size=(self.size,))
        self.t = 1

    def __check_t(self):
        if self.t >= len(self.timesteps):
            self.timesteps = torch.randint(low=0, high=self.noise_step, size=(self.size,))
            self.t = 1

    def sample(self):
        self.__check_t()
        value = self.timesteps[self.t - 1]
        self.t += 1
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
        self.t_sampler = TimeStepSampler(1000)
        self.loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)

    def training_step(self, batch, batch_idx):  
        #y = y_loss = batch.y
        mask = batch.get('mask')
        eval_mask = batch.get('eval_mask')
        
        t = self.t_sampler.sample()

        # Compute predictions and compute loss
        ε, ε_pred = self.predict_noise(batch, mask, t)

        # Compute loss
        loss = self.loss_fn(ε, ε_pred, eval_mask)

        # Logging
        #self.train_metrics.update(y_hat, y, mask)
        #self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #y = y_loss = batch.y
        mask = batch.get('mask')
        eval_mask = batch.get('eval_mask')
        
        t = self.t_sampler.sample()

        # Compute predictions and compute loss
        ε, ε_pred = self.predict_noise(batch, mask, t)

        # Compute loss
        loss = self.loss_fn(ε, ε_pred, eval_mask)

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
        imputation = self.sample()

        x_imputated = batch.x * mask + imputation * (1 - mask)

        # Compute loss
        loss = torch.zeros(1)

        # Logging
        self.test_metrics.update(x_imputated, x, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', loss, batch_size=batch.batch_size)
        return loss
