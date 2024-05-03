import os
import numpy as np
import pandas as pd
import numpy as np
from omegaconf import open_dict

from src.data.traffic import MetrLADataset
from src.experiments.experiment import Experiment

class MissingExperiment(Experiment):
    def __init__(self, **kwargs):
        self.p_fault, self.p_noise = kwargs['fault_noise']
        super().__init__(**kwargs)

    def prepare_data(self):
        dm_params = {
            'batch_size': self.cfg.config.batch_size,
            'scale_window_factor': self.cfg.config.scale_window_factor,
            'p_fault': self.p_fault,
            'p_noise': self.p_noise,
            'point': True if self.cfg['dataset']['scenario'] == 'point' else False
        }

        self.dm = MetrLADataset(**dm_params).get_dm()
        self.dm_stride = MetrLADataset(stride='window_size', **dm_params).get_dm()

        print(self.dm)

        with open_dict(self.cfg):
            self.cfg.config.time_steps = self.dm.window
            self.cfg.config.num_nodes = self.dm.n_nodes

        self.train_dataloader = self.dm.train_dataloader()
        self.val_dataloader = self.dm_stride.val_dataloader()
        self.test_dataloader = self.dm_stride.test_dataloader()

    def run(self, path):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()

        # Test
        self.model.load_model(path)
        self.model.freeze()

        results = self.trainer.test(self.model, self.test_dataloader)

        return results[0]

class MissingAverageExperiment:
    def __init__(self, dataset, cfg, optimizer_type, seed, epochs, accelerator='gpu', device=None, n=5):
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer_type = optimizer_type
        self.seed = seed
        self.epochs = epochs
        self.accelerator = accelerator
        self.device = device
        self.n = n
        self.folder = f'./metrics/'
        self.file_results = f'{self.folder}/results_by_rate.csv'
        self.file_average_results = f'{self.folder}/results_rate.csv'

        self.kwargs_experiment = {
            'dataset': self.dataset,
            'cfg': self.cfg,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'accelerator': self.accelerator,
            'device': self.device,
            'seed': seed,
        }

        self.missing_rates_point = { # rate: p_fault, p_noise
            0.1: (0, 0.0205),
            0.2: (0, 0.1295),
            0.3: (0, 0.2381),
            0.4: (0, 0.347),
            0.5: (0, 0.456),
            0.6: (0, 0.5649),
            0.7: (0, 0.6735),
            0.8: (0, 0.7822),
            0.9: (0, 0.8912),
        }

        self.missing_rates_block = { # rate: p_fault, p_noise
            0.1: (0.0007, 0),
            0.2: (0.002938, 0.05),
            0.3: (0.007516, 0.05),
            0.4: (0.012835, 0.05),
            0.5: (0.01923, 0.05),
            0.6: (0.027285, 0.05),
            0.7: (0.03786, 0.05),
            0.8: (0.05356, 0.05),
            0.9: (0.0819, 0.05),
        }

        self.init_result_folder()

    def init_result_folder(self):
        os.makedirs(self.folder, exist_ok=True)
        if len(os.listdir(self.folder)) == 0:
            results = pd.DataFrame(columns=[
                'mae', 
                'mse', 
                'mre',
                'rate',
                'id'
                ])
            results.to_csv(self.file_results)

    def save_results(self, results):
        results_df = pd.read_csv(self.file_results, index_col='Unnamed: 0')
        results_df.loc[results_df.shape[0]+1] = [
            results['test_mae'], 
            results['test_mse'], 
            results['test_mre'], 
            results['rate'],
            results['id']
            ]
        results_df.to_csv(self.file_results)

    def average_results(self):
        average_results = pd.DataFrame(columns=[
                'mae_mean',
                'mae_std',
                'mse_mean',
                'mse_std',
                'mre_mean',
                'mre_std',
                'rate'
            ])
        
        results_by_experiment = pd.read_csv(self.file_results, index_col='Unnamed: 0')
        
        average_results.loc[0] = [
            results_by_experiment['mae'].mean(),
            results_by_experiment['mae'].std(),
            results_by_experiment['mse'].mean(),
            results_by_experiment['mse'].std(),
            results_by_experiment['mre'].mean(),
            results_by_experiment['mre'].std(),
            results_by_experiment['rate'].mean(),
        ]

        average_results.to_csv(self.file_average_results)

    def run(self):
        for rate in np.arange(0.1, 1, 0.1):
            for i in range(self.n):
                results = pd.read_csv(self.file_results)
                done = results[(results['rate'] == rate) & (results['id'] == i)].shape[0] > 0
                if not done:
                    self.kwargs_experiment['seed'] = self.seed + i
                    self.kwargs_experiment['fault_noise'] = self.missing_rates_point[rate]

                    experiment = Experiment(**self.kwargs_experiment)
                    results = experiment.run()

                    results[0]['rate'] = rate
                    results[0]['id'] = i

                    self.save_results(results)

        self.average_results()