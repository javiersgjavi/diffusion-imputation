import os
import time
import random
import numpy as np
import pandas as pd
import numpy as np
from omegaconf import open_dict

import torch
from schedulefree import AdamWScheduleFree
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from tsl.metrics import torch as torch_metrics

from src.data.traffic import MetrLADataset, PemsBayDataset
from src.data.airquality import AQI36Dataset
from src.data.mimiciii import MimicIIIDataset
from src.models.diffusion import DiffusionImputer

class Experiment:
    def __init__(self, dataset, cfg, optimizer_type, epochs, accelerator='gpu', device=None, seed=42):
        self.cfg = cfg
        self.dataset = dataset
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.accelerator = accelerator
        self.device = device
        self.seed = seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def prepare_data(self):
        dm_params = {
            'batch_size': self.cfg.config.batch_size,
            'scale_window_factor': self.cfg.config.scale_window_factor
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

    def prepare_optimizer(self):
        if self.optimizer_type == 0:
            optimizer = Adam
            optimizer_kwargs = dict({'lr': 1e-3, 'weight_decay': 1e-6})

            p1 = int(0.75 * self.epochs)
            p2 = int(0.9 * self.epochs)

            scheduler = MultiStepLR
            scheduler_kwargs = {'milestones': [p1, p2], 'gamma': 0.1}

        elif self.optimizer_type == 1:
            steps_epoch = self.dm.train_len//self.dm.batch_size

            optimizer = AdamWScheduleFree
            #optimizer_kwargs = dict({'lr': 1e-2, 'weight_decay': 2e-6, 'warmup_steps': steps_epoch*5, 'betas': (0.9, 0.999), 'eps': 1e-8}) AQI 38
            #optimizer_kwargs = dict({'lr': 0.5e-2, 'weight_decay': 1e-6, 'warmup_steps': int(steps_epoch*0.75), 'betas': (0.95, 0.999), 'eps': 1e-8})
            optimizer_kwargs = dict({'lr':   5e-3, 'weight_decay': 0, 'warmup_steps': int(steps_epoch*0.75), 'betas': (0.98, 0.999), 'eps': 1e-8})

            scheduler = None
            scheduler_kwargs = None

        elif self.optimizer_type == 2:
            optimizer = Adam
            optimizer_kwargs = dict({'lr': 1e-3, 'weight_decay': 1e-6})

            p1 = int(0.75 * self.epochs)
            p2 = int(0.9 * self.epochs)

            steps_epoch = self.dm.train_len//self.dm.batch_size
            scheduler = CosineAnnealingLR
            scheduler_kwargs = {'T_max': steps_epoch}

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

    def prepare_model(self):
        
        cfg = dict(self.cfg)
        cfg['hist_patterns'] = self.hist_patterns

        self.model = DiffusionImputer(
            model_kwargs=cfg,
            optim_class=self.optimizer,
            optim_kwargs=self.optimizer_kwargs,
            # whiten_prob=list(np.arange(0,1,0.001)),
            whiten_prob=None,
            scheduler_class=self.scheduler,
            scheduler_kwargs=self.scheduler_kwargs,
            metrics = {
                'mae': torch_metrics.MaskedMAE(),
                'mse': torch_metrics.MaskedMSE(),
                'mre': torch_metrics.MaskedMRE()
            }
        )

        logger = TensorBoardLogger(
            save_dir='./logs',
        )

        self.callbacks = [
            ModelCheckpoint(
                monitor='val_loss',
                filename='{epoch}-{val_loss:.5f}',
                save_top_k=1,
                mode='min',
                verbose=True,
            )
        ]

        self.trainer = Trainer(
            max_epochs=self.epochs,
            default_root_dir='./logs',
            logger=logger,
            accelerator=self.accelerator,
            devices=[self.device] if self.device is not None else None,
            callbacks=self.callbacks,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=1
            )
        
    def run(self):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()

        # Train
        train_start = time.time()
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
        train_end = time.time()

        # Test
        
        self.model.load_model(self.callbacks[0].best_model_path)
        self.model.freeze()

        test_start = time.time()
        results = self.trainer.test(self.model, self.test_dataloader)
        test_end = time.time()

        training_time = train_end - train_start
        testing_time = test_end - test_start

        results[0]['training_time'] = training_time
        results[0]['testing_time'] = testing_time

        return results[0]

class AverageExperiment:
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

        self.kwargs_experiment = {
            'dataset': self.dataset,
            'cfg': self.cfg,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'accelerator': self.accelerator,
            'device': self.device,
            'seed': seed,
        }

        print(self.kwargs_experiment)
        self.init_result_folder()

    def init_result_folder(self):
        os.makedirs(self.folder, exist_ok=True)
        if len(os.listdir(self.folder)) == 0:
            results = pd.DataFrame(columns=[
                'mae', 
                'mse', 
                'mre', 
                'training_time', 
                'testing_time', 
                ])
            results.to_csv(f'{self.folder}/results_by_experiment.csv')

    def save_results(self, results, i):
        results_df = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')
        results_df.loc[i] = [
            results['test_mae'], 
            results['test_mse'], 
            results['test_mre'], 
            results['training_time'], 
            results['testing_time'], 
            ]
        results_df.to_csv(f'{self.folder}/results_by_experiment.csv')

    def average_results(self):
        average_results = pd.DataFrame(columns=[
                'mae_mean',
                'mae_std',
                'mse_mean',
                'mse_std',
                'mre_mean',
                'mre_std',
                'training_time_mean',
                'training_time_std',
                'testing_time_mean',
                'testing_time_std',
            ])
        
        results_by_experiment = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')
        
        average_results.loc[0] = [
            results_by_experiment['mae'].mean(),
            results_by_experiment['mae'].std(),
            results_by_experiment['mse'].mean(),
            results_by_experiment['mse'].std(),
            results_by_experiment['mre'].mean(),
            results_by_experiment['mre'].std(),
            results_by_experiment['training_time'].mean(),
            results_by_experiment['training_time'].std(),
            results_by_experiment['testing_time'].mean(),
            results_by_experiment['testing_time'].std(),
        ]

        average_results.to_csv(f'{self.folder}/results.csv')

    def run(self):
        n_done = pd.read_csv(f'{self.folder}/results_by_experiment.csv').shape[0]
        for i in range(n_done, self.n):
            self.kwargs_experiment['seed'] = self.seed + i
            experiment = Experiment(**self.kwargs_experiment)
            results = experiment.run()
            self.save_results(results, i)

        self.average_results()
        
