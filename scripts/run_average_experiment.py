import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.experiment import AverageExperiment

@hydra.main(config_name="base.yaml", config_path="../config")
def main(cfg: DictConfig):

    experiment = AverageExperiment(
        name_experiment='metr_la',
        dataset='metr-la',
        cfg=cfg,
        optimizer_type=0,
        seed=42,
        epochs=50,
        accelerator='gpu',
        device=0,
        n=5
    )

    experiment.run()

if __name__=='__main__':
    main()