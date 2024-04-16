import sys
import numpy as np
sys.path.append('./')

from torch.optim import Adam, AdamW
from src.data.traffic import MetrLADataset
from src.models.diffusion import DiffusionImputer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer


def main():
    
    dm = MetrLADataset().get_dm()
    dm_stride = MetrLADataset(stride=24).get_dm()

    dm.setup()
    dm_stride.setup()

    imputer = DiffusionImputer(
        model_kwargs=None,
        optim_class=AdamW,
        optim_kwargs=dict({'lr': 1e-3}),
        whiten_prob=list(np.arange(0,1,0.001)),
        prediction_loss_weight=1,
        impute_only_missing = True,
        warm_up_steps=1,
        scheduler_class=None,
        scheduler_kwargs=None,
    )

    logger = TensorBoardLogger(
        save_dir='./logs',
    )

    callbacks = [
        #EarlyStopping(
        #    monitor='val_loss',
        #    patience=200,
        #    verbose=True,
        #    mode='min'
        #),
        ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.5f}',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
    ]

    trainer = Trainer(
        max_epochs=50,
        default_root_dir='./logs',
        logger=logger,
        accelerator='gpu',
        devices=[2],
        callbacks=callbacks,
        #limit_train_batches=0.002
        )


    train_loader = dm.train_dataloader()
    val_loader = dm_stride.val_dataloader()
    trainer.fit(imputer, train_loader, val_loader)
    
    imputer.load_model(callbacks[0].best_model_path)
    imputer.freeze()

    trainer.test(imputer, datamodule=dm_stride)

if __name__=='__main__':
    main()
