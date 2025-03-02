"""  """

from __future__ import print_function, absolute_import, division

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import MSELoss

from src.models.components.outputs import AtomwiseV3
from src.models.tasks.Task import Task


class MD17Task(Task):
    name = "MD17"

    def __init__(self, representation, label_key, dataset_meta, task_config=None, **kwargs):
        task_defaults = {
            "loss_weights": [0.05, 0.95],
        }

        super().__init__(representation, label_key, dataset_meta, task_config=task_config, task_defaults=task_defaults,
                         **kwargs)
        self.num_classes = 1

    def get_losses(self):
        ema_rates = self.config.get('ema_rates', [0.05, 1.00])
        print(ema_rates, 'ema_rates', self.config['loss_weights'], 'loss_weights')
        if len(ema_rates) == 1:
            ema_rates.append(1.00)
        return [{
            "metric": MSELoss,
            "prediction": 'energy',
            "target": 'y',
            'ema_rate': ema_rates[0],
            "loss_weight": self.config['loss_weights'][0]
        }, {
            "metric": MSELoss,
            "prediction": 'force',
            "target": 'dy',
            'ema_rate': ema_rates[1],
            "loss_weight": self.config['loss_weights'][1]
        }]

    def get_metrics(self):
        return [
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": 'energy',
                "target": 'y',
            }, {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": 'energy',
                "target": 'y',
            },
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": 'force',
                "target": 'dy',
            }, {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": 'force',
                "target": 'dy',
            }
        ]

    def get_output(self, output_config=None):
        outputs = AtomwiseV3(
            n_in=self.representation.hidden_dim,
            mean=self.dataset_meta['mean'],
            stddev=self.dataset_meta['std'],
            atomref=None,
            aggregation_mode="sum",
            property="energy",
            activation=F.silu,
            derivative="force",
            negative_dr=True,
            create_graph=True,
            **output_config
        )
        outputs = [outputs]
        return torch.nn.ModuleList(outputs)

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ['test']
