import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.virtual_sensing_experiment import VirtualSensingExperimentAverage

@hydra.main(config_name="base.yaml", config_path="../config/virtual_sensing/")
def main(cfg: DictConfig):

    experiment = VirtualSensingExperimentAverage(
        dataset=cfg.dataset.name,
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