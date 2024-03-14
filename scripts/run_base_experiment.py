import sys
sys.path.append('./')

from torch.optim import Adam, AdamW
from src.data.traffic import MetrLADataset
from src.models.diffusion import DiffusionImputer

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer


def main():
    
    dm = MetrLADataset().get_dm()

    imputer = DiffusionImputer(
        model_kwargs=None,
        optim_class=AdamW,
        optim_kwargs=dict({'lr': 1e-3}),
        whiten_prob=None,
        prediction_loss_weight=1,
        impute_only_missing = True,
        warm_up_steps=1,
        scheduler_class=None,
        scheduler_kwargs=None,
    )

    logger = TensorBoardLogger(
        save_dir='./logs',
    )

    callbaks = [
        #EarlyStopping(
        #    monitor='val_loss',
        #    patience=200,
        #    verbose=True,
        #    mode='min'
        #),
        ModelCheckpoint(
            monitor='val_loss_epoch',
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
        devices=[0],
        #callbacks=callbaks,
        )

    trainer.fit(imputer, dm)
    
    trainer.test(imputer, datamodule=dm)#, ckpt_path=trainer.checkpoint_callback.best_model_path)

if __name__=='__main__':
    main()