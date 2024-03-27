import torch
from diffusers import DDPMScheduler
import numpy as np
import pandas as pd


def create_interpolation(data):
    x_interpolated = data['x']
    mask = data['mask']
    x_interpolated[mask == 0] = torch.nan
    '''x_interpolated = torchcde.linear_interpolation_coeffs(x_interpolated).permute(0,2,1,3)'''
    for i in range(x_interpolated.shape[0]):
        sample = x_interpolated[i].squeeze()
        df = pd.DataFrame(sample.cpu()).interpolate(method='linear', axis=0).backfill(axis=0).ffill(axis=0).fillna(0).values
        x_interpolated[i] = torch.tensor(df).unsqueeze(-1).to(x_interpolated.device)

    data['x_interpolated'] = x_interpolated
    return data

def redefine_eval_mask(data):
    og_mask = data['og_mask']
    cond_mask = data['mask']
    eval_mask = og_mask.int() - cond_mask
    data['eval_mask'] = eval_mask
    return data


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
    
class SchedulerBasic:
    def __init__(self, beta_start, beta_end, num_train_timesteps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = num_train_timesteps

        self.beta = torch.FloatTensor(np.linspace(self.beta_start  ** 0.5, self.beta_end ** 0.5, self.steps) ** 2)

        self.alpha_hat = 1 - self.beta
        self.alpha = torch.FloatTensor(np.cumprod(self.alpha_hat))
        self.alpha_torch = self.alpha
        self.alpha_hat = torch.FloatTensor(self.alpha_hat)

    def forward(self, observed_data, t):
        noise = torch.randn_like(observed_data).to(observed_data.device)
        t = t.to(self.alpha_torch.device).long()
        current_alpha = self.alpha_torch[t].to(observed_data.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        return noisy_data, noise
    
    def backwards(self, current_sample, predicted_noise, t):

        t_transform = t.to(self.alpha_torch.device).long()[0]

        coeff1 = 1 / self.alpha_hat[t_transform] ** 0.5
        coeff2 = (1 - self.alpha_hat[t_transform]) / (1 - self.alpha[t_transform]) ** 0.5

        current_sample = coeff1 * (current_sample - coeff2 * predicted_noise)

        if t[0] > 0:
            noise = torch.randn_like(current_sample).to(current_sample.device)
            sigma = (
                (1.0 - self.alpha[t_transform - 1]) / (1.0 - self.alpha[t_transform]) * self.beta[t_transform]
            ) ** 0.5

            current_sample += sigma.to(current_sample.device) * noise

        return current_sample


class SchedulerCSDI(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zeros = torch.zeros(self.size)

    def prepare_data(self, x_co_0, mask_co, u, t=None, x_real_0=None):

        if x_real_0 is None:
            noise = torch.randn(self.size)
            x_noisy_t = noise
        else:
            x_noisy_t, noise = self.forward(x_real_0, t)

        x_ta_t = torch.where(mask_co.bool(), self.zeros, x_noisy_t)
    
        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, self.size.shape[2], 1)
        cond_info = {
            'x_co': x_co_0,
            'mask_co': mask_co,
            'u': u,
        }

        return x_ta_t, cond_info, noise
    
    def clean_backwards(self, x_t, noise_pred, t, mask):
        x_t = self.backwards(x_t, noise_pred, t)
        x_t = torch.where(mask.bool(), self.zeros, x_t)
        return x_t    

class SchedulerPriSTI(SchedulerBasic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self, batch, t=None,):
        x_co_0 = batch.x
        x_co_0_itp = batch.x_interpolated
        x_real_0 = batch.transform['y'](batch.y)
        zeros = torch.zeros_like(x_co_0).to(x_co_0.device)
        x_real_0 = torch.where(batch.og_mask, x_real_0, zeros)
        mask_co = batch.mask
        u = batch.u

        if t is None:
            noise = torch.randn_like(x_co_0)
            x_noisy_t = noise
        else:
            x_noisy_t, noise = self.forward(x_real_0, t)

        x_ta_t = torch.where(mask_co.bool(), zeros, x_noisy_t)
    
        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, x_co_0.shape[2], 1)
        cond_info = {'x_co': x_co_0_itp, 'mask_co': mask_co,'u': u,}
        return x_ta_t, cond_info, noise
    
    def clean_backwards(self, x_t, noise_pred, mask, t):
        zeros = torch.zeros_like(x_t).to(x_t.device)
        x_t_1 = self.backwards(x_t, noise_pred, t)
        x_t = torch.where(mask.bool(), zeros, x_t_1)
        return x_t
