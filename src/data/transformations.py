from torch_geometric.transforms import BaseTransform
from tsl.data import Data
import pandas as pd
import numpy as np
import torch
import time
from tsl.data import ImputationDataset
from tsl.data import HORIZON
from tsl.data.batch_map import BatchMapItem


class CustomTransform(BaseTransform):
    def __init__(self, input_key='x', mask_key = 'mask', interpolated_key = 'x_interpolated'):
        self.input_key = input_key
        self.mask_key = mask_key
        self.interpolated_key = interpolated_key

    def __call__(self, data):
        data[self.input_key] *= data[self.mask_key]

        data.transform[self.input_key](data[self.interpolated_key])
        return data
    
class ImputatedDataset(ImputationDataset):
    def __init__(self, x_interpolated, **kwargs):
        super().__init__(**kwargs)
        self.add_covariate(
            name='x_interpolated',
            value=x_interpolated,
            pattern='t n f',
            add_to_input_map=True,  # NB
            synch_mode=HORIZON,
            preprocess=False) # quizás hay que cambiarlo

        self.auxiliary_map['x_interpolated'] = BatchMapItem('x_interpolated',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=False)

   