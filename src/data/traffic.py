import numpy as np
import pandas as pd
from tsl.datasets import MetrLA, PemsBay
from tsl.data import ImputationDataset
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.data.preprocessing import  MinMaxScaler, StandardScaler
from tsl.data.datamodule import TemporalSplitter, SpatioTemporalDataModule

from src.data.transformations import ImputatedDataset, CustomTransform, CustomScaler

class TrafficDataset:
    def __init__(self, p_fault=None, p_noise=None, point=True, stride=1):

        og_mask = self.data_class(impute_zeros=False).mask

        if point:
            p_fault = 0. if p_fault is None else p_fault
            p_noise = 0.25 if p_noise is None else p_noise

        if not point:
            p_fault = 0.0015 if p_fault is None else p_fault
            p_noise = 0.05 if p_noise is None else p_noise

        dataset = add_missing_values(
            self.data_class(impute_zeros=False),
            p_fault=p_fault,
            p_noise=p_noise,
            min_seq=12,
            max_seq=12 * 4,
            seed=self.seed
        )

        connectivity = dataset.get_connectivity(
            threshold=0.1,
            include_self=False,
            normalize_axis=1,
            layout="edge_index"
            )

        covariates = {'u': dataset.datetime_encoded('day').values}
        
        torch_dataset = ImputatedDataset(
            target=dataset.dataframe(),
            og_mask = og_mask,
            mask = dataset.training_mask,
            eval_mask = dataset.eval_mask,
            covariates=covariates,
            transform=CustomTransform(),
            connectivity=connectivity,
            window=24*4,
            stride=stride
        )

        print(f'Real missing values: {np.round((1 - np.mean(dataset.mask))* 100, 2)} %')
        print(f'Final missing values: {np.round((1 - np.mean(dataset.training_mask))* 100, 2)} %')

        scalers = {'target': CustomScaler()}
       
        splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=4,
            )

    def get_dm(self):
        return self.dm

class MetrLADataset(TrafficDataset):
    def __init__(self, **kwargs):
        self.data_class = MetrLA
        self.seed = 9101112
        self.dataset= 'la'
        super().__init__(**kwargs)

class PemsBayDataset(TrafficDataset):
    def __init__(self, **kwargs):
        self.data_class = PemsBay
        self.seed = 9101112
        self.dataset='bay'
        super().__init__(**kwargs)