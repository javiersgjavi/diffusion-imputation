import os
import sys
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


sys.path.append('./')

from torch.optim import Adam, AdamW

from tsl.metrics import torch as torch_metrics
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from schedulefree import AdamWScheduleFree
from src.data.traffic import MetrLADataset
from src.models.diffusion import DiffusionImputer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

@hydra.main(config_name="base.yaml", config_path="../config")
def main(cfg: DictConfig):

    epochs = 50
    optimizer_type = 0

    dm_params = {
        'batch_size': cfg.config.batch_size,
        'scale_window_factor': cfg.config.scale_window_factor
    }

    dm = MetrLADataset(**dm_params).get_dm()
    dm_stride = MetrLADataset(stride='window_size', **dm_params).get_dm()

    dm.setup()
    dm_stride.setup()
    
    if optimizer_type == 0:
        optimizer = Adam
        optimizer_kwargs = dict({'lr': 1e-3, 'weight_decay': 1e-6})

        p1 = int(0.75 * epochs)
        p2 = int(0.9 * epochs)

        scheduler = MultiStepLR
        scheduler_kwargs = {'milestones': [p1, p2], 'gamma': 0.1}

    elif optimizer_type == 1:
        steps_epoch = dm.train_len//dm.batch_size

        optimizer = AdamWScheduleFree
        optimizer_kwargs = dict({'lr': 1e-2, 'weight_decay': 1e-6, 'warmup_steps': steps_epoch*10, 'betas': (0.95, 0.99), 'eps': 1e-9})

        scheduler = None
        scheduler_kwargs = None

    elif optimizer_type == 2:
        optimizer = Adam
        optimizer_kwargs = dict({'lr': 1e-3, 'weight_decay': 1e-6})

        p1 = int(0.75 * epochs)
        p2 = int(0.9 * epochs)

        steps_epoch = dm.train_len//dm.batch_size
        scheduler = CosineAnnealingLR
        scheduler_kwargs = {'T_max': steps_epoch}


    with open_dict(cfg):
        cfg.config.time_steps = dm.window
        cfg.config.num_nodes = dm.n_nodes

    imputer = DiffusionImputer(
        model_kwargs=dict(cfg),
        optim_class=optimizer,
        optim_kwargs=optimizer_kwargs,
        whiten_prob=list(np.arange(0,1,0.001)),
        scheduler_class=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        metrics = {
            'mae': torch_metrics.MaskedMAE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
    )

    logger = TensorBoardLogger(
        save_dir='./logs',
    )

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.5f}',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
    ]

    trainer = Trainer(
        max_epochs=epochs,
        default_root_dir='./logs',
        logger=logger,
        accelerator='gpu',
        #devices=[2],
        callbacks=callbacks,
        )


    train_loader = dm.train_dataloader()
    val_loader = dm_stride.val_dataloader()
    trainer.fit(imputer, train_loader, val_loader)
    
    imputer.load_model(callbacks[0].best_model_path)
    imputer.freeze()

    trainer.test(imputer, datamodule=dm_stride)

if __name__=='__main__':
    main()
