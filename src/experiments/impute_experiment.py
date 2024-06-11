import os
import torch
import numpy as np
from tqdm import tqdm

from src.experiments.experiment import Experiment

class ImputeExperiment(Experiment):
    def __init__(self, weights_path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = f'../../imputed_data/{self.dataset}/{self.cfg.model_name}/'

    def clear_data(self, data):
        data = torch.cat(data, dim=0).cpu().detach()
        if self.cfg.dataset.name == 'mimic-iii':
            data = data.squeeze().permute(0, 2, 1)
        return data.numpy()
    
    def impute_dataloader(self, dataloader, name):
        imputed_samples = []
        for batch in tqdm(dataloader, desc=f'Imputing {name} data'):
            batch = batch.to(self.device)
            batch_imputed = self.model.predict_step(batch, 0)
            imputed_samples.append(batch_imputed)
        imputed_samples = self.clear_data(imputed_samples)
        np.save(f'{self.save_path}{name}_samples.npy', imputed_samples)
        
    def run(self):
        self.cfg.config.batch_size = 1
        os.makedirs(self.save_path, exist_ok=True)
        
        self.prepare_data()
        self.prepare_optimizer()
        self.prepare_model()
        
        self.model.load_model(self.cfg.weights.path)
        self.model.freeze()

        self.train_dataloader = self.dm.train_dataloader(shuffle=False)

        self.impute_dataloader(self.test_dataloader, 'test')
        self.impute_dataloader(self.val_dataloader, 'val')
        self.impute_dataloader(self.train_dataloader, 'train')
