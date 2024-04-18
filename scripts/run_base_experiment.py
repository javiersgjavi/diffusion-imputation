import sys
import numpy as np
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

def main():

    epochs = 50
    optimizer_type = 0

    dm = MetrLADataset().get_dm()
    dm_stride = MetrLADataset(stride=24).get_dm()

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

    model_kwargs = {
        # EMA hyperparameters:
        'use_ema': False,
        'decay': 0.995,

        # Scheduler hyperparameters:
        'scheduler_kwargs':{
            'num_train_timesteps':50,
            'beta_schedule':'scaled_linear',
            'beta_start':0.0001,
            'beta_end':0.2,
            'clip_sample':False
            },

        # Model hyperparameters:
        'config' : {
            
            # Base PriSTI hyperparameters
            'layers': 4,
            'channels': 64, 
            'nheads': 8, 
            'diffusion_embedding_dim': 128, 
            'schedule': 'quad', 
            'is_adp': True, 
            'proj_t': 64, 
            'is_cross_t': True, 
            'is_cross_s': True, 
            'adj_file': 'metr-la', 
            'side_dim': 144,
            'num_nodes': dm.n_nodes,
            'time_steps': dm.window,
            'batch_size': dm.batch_size,
        }

    }


    imputer = DiffusionImputer(
        model_kwargs=model_kwargs,
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
        devices=[2],
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
