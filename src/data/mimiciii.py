
import torch
import numpy as np
import pandas as pd

from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.ops.imputation import add_missing_values
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.datasets.prototypes.tabular_dataset import TabularDataset

from src.data.transformations import ImputatedDataset, CustomTransform, CustomScaler

class MimicIIIDataset:
    def __init__(self, batch_size=4, p_fault=0., p_noise=0.25, **kwargs):

        self.seed = 42
        self.window_size = 49

        train_data = np.load('../../data/mimic-iii/train/x_dl.npy')
        val_data = np.load('../../data/mimic-iii/val/x_dl.npy')
        test_data = np.load('../../data/mimic-iii/test/x_dl.npy')

        data = np.concatenate([train_data, val_data, test_data], axis=0)
        data = np.transpose(data, (0, 2, 1)).astype(np.float32).reshape(-1, data.shape[1])
        data_pd = pd.DataFrame(data)

        og_mask = ~np.isnan(data).astype(bool).reshape(data.shape[0], data.shape[1], 1)

        tabular_dataset = TabularDataset(
            target=data_pd,
            mask=og_mask,
        )

        dataset = add_missing_values(
            tabular_dataset,
            p_fault=p_fault,
            p_noise=p_noise,
            min_seq=self.window_size,
            max_seq=self.window_size * 2,
            seed=self.seed
        )

        connectivity = self._calculate_connectivity(data_pd)
        covariates = {'u': np.zeros((data_pd.shape[0], 2))}

        data[np.isnan(data).astype(bool)] = 0

        
        torch_dataset = ImputatedDataset(
            target=data.reshape(data.shape[0], data.shape[1], 1),
            og_mask=og_mask,
            mask=dataset.training_mask,
            eval_mask=dataset.eval_mask,
            covariates=covariates,
            transform=CustomTransform(),
            connectivity=connectivity,
            window=self.window_size,
            stride=self.window_size
        )

        print(f'Real missing values: {np.round((1 - np.mean(dataset.mask))* 100, 2)} %')
        print(f'Final missing values: {np.round((1 - np.mean(dataset.training_mask))* 100, 2)} %')

        scalers = {'target': CustomScaler()}


        idx_train_set = train_data.shape[0]
        idx_val_set = idx_train_set + val_data.shape[0]
        idx_test_set = idx_val_set + test_data.shape[0]


        splitter = FixedIndicesSplitter(
            train_idxs=[i for i in range(idx_train_set)],
            val_idxs=[i for i in range(idx_train_set, idx_val_set)],
            test_idxs=[i for i in range(idx_val_set, idx_test_set)]
        )

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=batch_size,
            )
        
    def get_dm(self):
        return self.dm
    
    def _calculate_connectivity(self, data_pd):

        n_vars = data_pd.shape[1]

        edge_index = np.array([
            np.repeat(np.arange(n_vars), n_vars),
            np.tile(np.arange(n_vars), n_vars)
        ])

        correlation = data_pd.corr()

        edge_weights = []
        for i in range(edge_index.shape[1]):
            e1 = edge_index[0, i]
            e2 = edge_index[1, i]
            value = correlation.iloc[e1, e2]
            edge_weights.append(value)

        edge_weights = np.array(edge_weights).astype(np.float32)

        matrix = np.zeros((n_vars, n_vars))
        for i in range(edge_weights.shape[0]):
            matrix[edge_index[0, i], edge_index[1, i]] = edge_weights[i]

        np.save('../../data/mimic-iii/mimic_graph.npy', matrix)

        return edge_index, edge_weights
    
    def get_historical_patterns(self):
        self.dm.setup()
        train_loader = self.dm.train_dataloader()
        historical_patterns = []
        for batch in train_loader:
            historical_patterns.append(batch.mask)

        return torch.cat(historical_patterns, dim=0)