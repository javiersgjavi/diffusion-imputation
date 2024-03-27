from torch_geometric.transforms import BaseTransform
import torch
from tsl.data import ImputationDataset
from tsl.data import HORIZON
from tsl.data.batch_map import BatchMapItem
from tsl.data.preprocessing import  StandardScaler, Scaler, ScalerModule
import torchcde
import pickle
import time
import pandas as pd
import numpy as np




class CustomTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.nan = np.nan

    def normalize(self, data, label):
        data[label] = data.transform['x'](data[label])
        return data
    
    def mask_input(self, data):
        data['x'] *= data['mask']
        return data

    def mask_y(self, data):
        data['y'] = torch.where(data['og_mask'], data.target['y'], 0)
        return data

    
    
    
    def __call__(self, data):
        #print('mask data')
        data = self.mask_input(data)
        data = self.mask_y(data)
        #print(f'Interpolation time: {time.time() - t}')
        return data
    
class ImputatedDataset(ImputationDataset):
    def __init__(self, og_mask, **kwargs):
        super().__init__(**kwargs)
        self.add_covariate(
            name='x_interpolated',
            value=torch.zeros(og_mask.shape),
            pattern='t n f',
            add_to_input_map=True,  # NB
            synch_mode=HORIZON,
            preprocess=False) # quizás hay que cambiarlo
        
        self.add_covariate(
            name='og_mask',
            value=og_mask,
            pattern='t n f',
            add_to_input_map=True,  # NB
            synch_mode=HORIZON,
            preprocess=False) # quizás hay que cambiarlo

        self.auxiliary_map['x_interpolated'] = BatchMapItem('x_interpolated',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=True)
        
        self.auxiliary_map['og_mask'] = BatchMapItem('og_mask',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=False)

class CustomScaler(Scaler):
    def __init__(self, axis=0):
        super().__init__()
        with open('./data/metr_la/metr_meanstd.pk', 'rb') as f:
            self.bias, self.scale = pickle.load(f)

        self.bias = torch.tensor(self.bias, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        self.scale = torch.tensor(self.scale, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    def fit(self, *args, **kwargs):
        return self
    