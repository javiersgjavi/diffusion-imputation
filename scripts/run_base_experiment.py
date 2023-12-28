import sys
sys.path.append('./')

from torch.optim import Adam
from src.data.traffic import MetrLADataset
from src.model.engine import CustomImputer
from src.model.diffusion_model import DiffusionModel

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer


def main():
    dm = MetrLADataset()
    imputer = CustomImputer(
        model_class= DiffusionModel,
        model_kwargs=None,
        optim_class=Adam,
        optim_kwargs=dict({'lr': 1e-3}),
        whiten_prob=None,
        prediction_loss_weight=1,
        impute_only_missing = True,
        warm_up_steps=1,
        scheduler_class=None,
        scheduler_kwargs=None,
    )

    logger = WandbLogger(
        name='base_experiment',
        offline=True,
        project='tsl',
    )

    callbaks = [
        EarlyStopping(
            monitor='val_mse',
            patience=2,
            verbose=True,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_mse',
            filename='base_experiment',
            save_top_k=1,
            mode='min'
        )
    ]

    trainer = Trainer(
        max_epochs=10,
        default_root_dir='./logs',
        logger=logger,
        accelerator='gpu',
        devices=1,
        callbacks=callbaks,
        )
    
    trainer.fit(imputer, dm)


if __name__=='__main__':
    main()