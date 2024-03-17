import torch
from diffusers import DDPMScheduler


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

class SchedulerPriSTI(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self, batch, t=None,):
        x_co_0 = batch.x
        x_co_0_itp = batch.x_interpolated
        x_real_0 = batch.transform['y'](batch.y)
        mask_co = batch.mask
        u = batch.u

        if t is None:
            noise = torch.randn_like(x_co_0)
            x_noisy_t = noise
        else:
            x_noisy_t, noise = self.forward(x_real_0, t)

        x_ta_t = torch.where(mask_co.bool(), x_co_0, x_noisy_t)
    
        u = u.view(u.shape[0], u.shape[1], 1, u.shape[2]).repeat(1, 1, x_co_0.shape[2], 1)
        cond_info = {'x_co': x_co_0_itp, 'mask_co': mask_co,'u': u,}
        return x_ta_t, cond_info, noise
    
    def clean_backwards(self, x_t, noise_pred, mask, t):
        x_t_1 = self.backwards(x_t, noise_pred, t)
        x_t = torch.where(mask.bool(), x_t, x_t_1)
        return x_t
