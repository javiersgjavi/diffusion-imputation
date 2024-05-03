import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.experiment import AverageExperiment

@hydra.main(config_name="metr-la_point.yaml", config_path="../config")
def main(cfg: DictConfig):

    weights_file = ''
    experiment = MissingAverageExperiment(
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