import torch
from diffusers import DDPMScheduler
import numpy as np
import pandas as pd


def create_interpolation(data):
    x_interpolated = data['x']
    mask = data['mask']
    x_interpolated[mask == 0] = torch.nan

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
        step = step.int()
        return super().add_noise(x, noise, step), noise
    
    def backwards(self, x_t, predicted_noise, step):
        step = step[0].int()
        return super().step(predicted_noise, step, x_t).prev_sample

class SchedulerPriSTI(Scheduler):
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
