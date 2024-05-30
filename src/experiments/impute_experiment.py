import torch
import numpy as np
from tqdm import tqdm

from src.experiments.experiment import Experiment

class ImputeExperiment(Experiment):
    def __init__(self, weights_path, **kwargs):
        super().__init__(**kwargs)
        self.weights_path = weights_path

        self.save_path = f'../../imputed_data/{self.dataset}/{self.cfg.model_name}'
        
    def run(self):

        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()
        
        self.model.load_model(self.weights_path)
        self.model.freeze()

        train_samples = []
        val_samples = []
        test_samples = []

        self.train_dataloader = self.dm.train_dataloader(shuffle=False)
        print(self.train_dataloader)

        for batch in tqdm(iter(self.train_dataloader), desc='Imputing training data'):
            batch.to(self.device)
            batch_imputed = self.model.predict_step(batch, 0)
            train_samples.append(batch)
        
        for batch in tqdm(iter(self.val_dataloader), desc='Imputing validation data'):
            batch.to(self.device)
            batch_imputed = self.model.predict_step(batch, 0)
            val_samples.append(batch)

        for batch in tqdm(iter(self.test_dataloader), desc='Imputing test data'):
            batch.to(self.device)
            batch_imputed = self.model.predict_step(batch, 0)
            test_samples.append(batch)

        train_samples = torch.cat(train_samples, dim=0)
        val_samples = torch.cat(val_samples, dim=0)
        test_samples = torch.cat(test_samples, dim=0)

        np.save(f'{self.save_path}train_samples.npy', train_samples.numpy())
        np.save(f'{self.save_path}val_samples.npy', val_samples.numpy())
        np.save(f'{self.save_path}test_samples.npy', test_samples.numpy())



