import torch
from diffusers import DDPMScheduler
import pandas as pd
from tsl.ops.imputation import sample_mask



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
    def __init__(self, high=1, low=0, size=1e6, dtype_int=False):
        self.size = int(size)
        self.high = high
        self.low = low
        self.fn = self.generate_int if dtype_int else self.generate_float
        self.stack = self.fn()
        self.idx = 0

    def generate_int(self):
        return torch.randint(low=self.low, high=self.high, size=(self.size,))
    
    def generate_float(self):
        return torch.FloatTensor(self.size).uniform_(self.low, self.high)

    def get(self, n=1):
        if self.idx + n >= self.size:
            self.stack = self.fn()
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
    

class PointStrategy:
    def __init__(self):
        print('PointStrategy')
        self.rnd_stack = RandomStack()

    def get_mask(self, batch):
        missing_rate = self.rnd_stack.get(batch.x.shape[0]).to(batch.x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        custom_mask = torch.rand_like(batch.x).to(batch.x.device) < missing_rate
        return custom_mask
    
class BlockStrategy:
    def __init__(self, p=0.15, p_noise=0.05, min_seq=None, max_seq=24):
        self.p_stack = RandomStack(high=p)
        self.p_noise = p_noise
        self.max_seq = max_seq
        self.min_seq = min_seq if min_seq is not None else max_seq // 2
        print(f'BlockStrategy: p={p}, p_noise={self.p_noise}, min_seq={self.min_seq}, max_seq={self.max_seq}')

    def get_mask(self, batch):
        sample_size = batch.x.shape[1:]
        custom_mask = torch.zeros_like(batch.x).to(batch.x.device)
        for i in range(custom_mask.shape[0]):
            p = self.p_stack.get().item()
            
            custom_mask_sample = sample_mask(
                sample_size, 
                p, 
                self.p_noise, 
                self.min_seq, 
                self.max_seq,
                verbose=False
                )
            custom_mask[i] = torch.tensor(custom_mask_sample).to(custom_mask.device)
        return custom_mask

class HistoryStrategy:
    def __init__(self, hist_patterns):
        self.hist_patterns = hist_patterns
        print(f'HistoryStrategy with {self.hist_patterns.shape[0]} patterns')
        self.rnd_stack_int = RandomStack(self.hist_patterns.shape[0], dtype_int=True)

    def get_mask(self, batch):
        index = self.rnd_stack_int.get(batch.x.shape[0])
        patterns = self.hist_patterns[index].to(batch.x.device)
        return patterns
    

class MissingPatternHandler:
    def __init__(self, strategy1='point', strategy2=None, hist_patterns=None, seq_len=None):
        self.strategy1 = self.get_class_strategy(strategy1, hist_patterns, seq_len)
        self.strategy2 = self.get_class_strategy(strategy2, hist_patterns, seq_len)
        self.rnd_stack = RandomStack()


    def get_class_strategy(self, strategy, hist_patterns, seq_len=None):
        if strategy == 'point':
            return PointStrategy()
        elif strategy == 'block':
            return BlockStrategy(max_seq=seq_len)
        elif strategy == 'historical':
            return HistoryStrategy(hist_patterns)
        else:
            return None

    def update_mask(self, batch):
        custom_mask = self.strategy1.get_mask(batch).bool()
        new_mask = batch.mask & custom_mask

        if self.strategy2 is not None:
            custom_mask2 = self.strategy2.get_mask(batch).bool()
            new_mask2 = batch.mask & custom_mask2

            mask_sample = self.rnd_stack.get(batch.x.shape[0]).to(batch.x.device) < 0.5
            for i, use_mask_2 in enumerate(mask_sample):
                if use_mask_2:
                    new_mask[i] = new_mask2[i]

        batch.mask = new_mask
        batch.input.x = batch.input.x * batch.mask

    def check_missing_rate(self, batch):
        missing_rate = 1 - torch.mean(batch.mask.float())
        return torch.round(missing_rate, decimals=4)


