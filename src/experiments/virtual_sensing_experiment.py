import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from omegaconf import open_dict


from src.data.airquality import AQI36Dataset
from src.data.mimiciii import MimicIIIDataset
from src.data.traffic import MetrLADataset, PemsBayDataset

from src.experiments.experiment import Experiment, AverageExperiment

class VirtualSensingExperiment(Experiment):

    def prepare_data(self):
        self.masked_sensors = self.cfg['dataset']['masked_sensors']
        dm_params = {
            'batch_size': self.cfg.config.batch_size,
            'scale_window_factor': self.cfg.config.scale_window_factor,
            'masked_sensors': self.masked_sensors
        }

        if self.dataset == 'metr-la':
            data_class = MetrLADataset
            dm_params['point'] = True if self.cfg['dataset']['scenario'] == 'point' else False

        elif self.dataset == 'pems-bay':
            data_class = PemsBayDataset
            dm_params['point'] = True if self.cfg['dataset']['scenario'] == 'point' else False

        elif self.dataset == 'aqi-36':
            data_class = AQI36Dataset

        elif self.dataset == 'mimic-iii':
            data_class = MimicIIIDataset

        self.dm = data_class(**dm_params).get_dm()
        self.dm_stride = data_class(stride='window_size', **dm_params).get_dm()

        if self.cfg.missing_pattern.strategy1 == 'historical' or self.cfg.missing_pattern.strategy2 == 'historical':
            # Use only historical patterns for the same months as the one selected in CSDI and PriSTI
            self.hist_patterns = data_class(test_months=(2, 5, 8, 11), **dm_params).get_historical_patterns()
        else:
            self.hist_patterns = None

        self.dm.setup()
        self.dm_stride.setup()

        print(self.dm)

        with open_dict(self.cfg):
            self.cfg.config.time_steps = self.dm.window
            self.cfg.config.num_nodes = self.dm.n_nodes

        self.train_dataloader = self.dm.train_dataloader()
        self.val_dataloader = self.dm_stride.val_dataloader()
        self.test_dataloader = self.dm_stride.test_dataloader()

    def run(self):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()

        # Train
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

        # Test
        self.model.load_model(self.callbacks[0].best_model_path)
        self.model.freeze()

        dict_sensors = {i:[] for i in self.masked_sensors}
        res = {
            'mae':dict_sensors,
            'mse':deepcopy(dict_sensors)
            }
        
        self.model = self.model.to(self.device)
        
        for batch in tqdm(self.test_dataloader):
            batch = batch.to(self.device)
            results = self.model.test_step_virtual_sensing(batch, self.masked_sensors)
            for sensor in self.masked_sensors:
                res['mae'][sensor].append(results['mae'][sensor])
                res['mse'][sensor].append(results['mse'][sensor])

        for sensor in self.masked_sensors:
            res['mae'][sensor] = np.mean(res['mae'][sensor])
            res['mse'][sensor] = np.mean(res['mse'][sensor])

        return res

class VirtualSensingExperimentAverage(AverageExperiment):
    def __init__(self, **kwargs):
        cfg = kwargs['cfg']
        self.masked_sensors = cfg['dataset']['masked_sensors']
        self.columns = [f'mae_{sensor}' for sensor in self.masked_sensors] + [f'mse_{sensor}' for sensor in self.masked_sensors]
        super().__init__(**kwargs)

    def init_result_folder(self):
        os.makedirs(self.folder, exist_ok=True)
        if len(os.listdir(self.folder)) == 0:
            results = pd.DataFrame(columns=self.columns)
            results.to_csv(f'{self.folder}/results_by_experiment.csv')

    def save_results(self, results, i):
        results_df = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')
        row = []
        for column in self.columns:
            metric, sensor = column.split('_')
            row.append(results[metric][int(sensor)])

        results_df.loc[i] = row
        results_df.to_csv(f'{self.folder}/results_by_experiment.csv')

    def average_results(self):
        columns_average = [f'{column}_mean' for column in self.columns] + [f'{column}_std' for column in self.columns]
        average_results = pd.DataFrame(columns=columns_average)
        
        results_by_experiment = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')

        res = []
        for column in columns_average:
            metric, sensor, stat = column.split('_')
            if stat == 'mean':
                res.append(results_by_experiment[f'{metric}_{sensor}'].mean())
            elif stat == 'std':
                res.append(results_by_experiment[f'{metric}_{sensor}'].std())

        average_results.loc[0] = res
        average_results.to_csv(f'{self.folder}/results.csv')

    def run(self):
        n_done = pd.read_csv(f'{self.folder}/results_by_experiment.csv').shape[0]
        for i in range(n_done, self.n):
            self.kwargs_experiment['seed'] = self.seed + i
            experiment = VirtualSensingExperiment(**self.kwargs_experiment)
            results = experiment.run()
            self.save_results(results, i)

        self.average_results() 