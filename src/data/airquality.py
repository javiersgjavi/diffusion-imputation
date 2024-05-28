import torch
import numpy as np

from typing import List, Optional, Sequence

from tsl.datasets.air_quality import AirQuality, AirQualitySplitter
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.data.synch_mode import HORIZON
from tsl.data.datamodule.splitters import disjoint_months


from src.data.transformations import ImputatedDataset, CustomTransform, CustomScaler

class AirDatasetSplitter(AirQualitySplitter):
    
    def __init__(self,
                 val_len: int = 0.1,
                 val_months: Sequence = (2, 5, 8, 11),
                 test_months: Sequence = (3, 6, 9, 12)):
        super(AirQualitySplitter, self).__init__()
        self._val_len = val_len
        self.val_months = val_months
        self.test_months = test_months


    def fit(self, dataset):

        # get indices of training, validation and testing months
        nontest_idxs, test_idxs = disjoint_months(
            dataset,
            months=self.test_months,
            synch_mode=HORIZON
        )

        # get index of validation months
        _, val_idxs = disjoint_months(
            dataset,
            months=self.val_months,
            synch_mode=HORIZON
        )

        # find index of each month of the validation set
        delta = np.diff(val_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        idx_sets = np.split(val_idxs, delta_idxs + 1)

        # get the last 10% of each month
        final_idxs = []
        for idx_set in idx_sets:
            idx_last = int(len(idx_set) * 0.1)
            final_idxs.extend(idx_set[-idx_last:])

        final_idxs_val = np.array(final_idxs)

        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs,
                                                  final_idxs_val,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]

        self.set_indices(train_idxs, final_idxs_val, test_idxs)


class AirDataset:
    def __init__(self, stride=1, batch_size=4, scale_window_factor=1, test_months=(3, 6, 9, 12)):
        self.base_window = 36

        self.window_size = int(self.base_window * scale_window_factor)
        stride = self.window_size if stride == 'window_size' else stride

        og_mask = AirQuality(small=self.small, test_months=test_months).mask
        dataset = AirQuality(small=self.small, test_months=test_months)

        connectivity = dataset.get_connectivity(
            include_self=False,
            normalize_axis=1,
            layout="edge_index"
            )
        
        covariates = {'u': dataset.datetime_encoded('day').values}
        
        torch_dataset = ImputatedDataset(
            target=dataset.dataframe(),
            og_mask=og_mask,
            mask=dataset.training_mask,
            eval_mask=dataset.eval_mask,
            covariates=covariates,
            transform=CustomTransform(),
            connectivity=connectivity,
            window=self.window_size,
            stride=stride
        )

        print(f'Real missing values: {np.round((1 - np.mean(dataset.mask))* 100, 2)} %')
        print(f'Final missing values: {np.round((1 - np.mean(dataset.training_mask))* 100, 2)} %')

        scalers = {'target': CustomScaler()}

        #splitter = dataset.get_splitter(method='air_quality', val_len=0.1)

        splitter = AirDatasetSplitter()

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=batch_size,
            )
        
    def get_dm(self):
        return self.dm
    
    def get_historical_patterns(self):
        self.dm.setup()
        test_loader = self.dm.test_dataloader()
        historical_patterns = []
        for batch in test_loader:
            historical_patterns.append(batch.mask)

        return torch.cat(historical_patterns, dim=0)


class AQI36Dataset(AirDataset):
    def __init__(self, **kwargs):
        self.small = True
        super().__init__(**kwargs)