from tsl.engines.imputer import Imputer
from tsl.metrics import torch as torch_metrics

class CustomImputer(Imputer):
    def __init__(self, *args, **kwargs):
        kwargs['loss_fn'] = torch_metrics.MaskedMSE(compute_on_step=True)
        kwargs['metrics'] = {
            'mae': torch_metrics.MaskedMAE(),
            #'rmse': torch_metrics.MaskedRMSE(),
            'mse': torch_metrics.MaskedMSE(),
            'mre': torch_metrics.MaskedMRE()
        }
        super().__init__(*args, **kwargs)

        print(self.optim_class,self.optim_kwargs)
        print(type(self.optim_kwargs))