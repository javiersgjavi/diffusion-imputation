import sys
import hydra
from omegaconf import DictConfig

sys.path.append('./')
from src.experiments.test_experiment import TestExperiment

@hydra.main(config_name="base.yaml", config_path="../config/test/")
def main(cfg: DictConfig):

    experiment = TestExperiment(
        dataset=cfg.dataset.name,
        cfg=cfg,
        optimizer_type=0,
        seed=42,
        epochs=50,
        accelerator='gpu',
        device=0,
    )

    results = experiment.run()
    print(results)

if __name__=='__main__':
    main()