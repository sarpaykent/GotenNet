"""MD17 task implementation for molecular dynamics simulations."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import MSELoss

from gotennet.models.components.outputs import AtomwiseV3
from gotennet.models.tasks.Task import Task


class MDTask(Task):
    """
    Task for MD17 molecular dynamics dataset.
    
    This task predicts energy and forces for molecular dynamics simulations.
    """

    name = "MD17"

    def __init__(
        self,
        representation,
        label_key,
        dataset_meta,
        task_config=None,
        **kwargs
    ):
        """
        Initialize the MD17 task.
        
        Args:
            representation: The representation model to use.
            label_key: The key for the label in the dataset.
            dataset_meta: Metadata about the dataset.
            task_config (dict, optional): Configuration for the task. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        task_defaults = {
            "loss_weights": [0.05, 0.95],
        }

        super().__init__(
            representation,
            label_key,
            dataset_meta,
            task_config=task_config,
            task_defaults=task_defaults,
            **kwargs
        )
        self.num_classes = 1

    def get_losses(self):
        """
        Get the loss functions for the MD17 task.
        
        Returns:
            list: A list of dictionaries containing loss function configurations for energy and force.
        """
        ema_rates = self.config.get('ema_rates', [0.05, 1.00])
        print(ema_rates, 'ema_rates', self.config['loss_weights'], 'loss_weights')
        if len(ema_rates) == 1:
            ema_rates.append(1.00)

        return [
            {
                "metric": MSELoss,
                "prediction": 'energy',
                "target": 'y',
                'ema_rate': ema_rates[0],
                "loss_weight": self.config['loss_weights'][0]
            },
            {
                "metric": MSELoss,
                "prediction": 'force',
                "target": 'dy',
                'ema_rate': ema_rates[1],
                "loss_weight": self.config['loss_weights'][1]
            }
        ]

    def get_metrics(self):
        """
        Get the metrics for the MD17 task.
        
        Returns:
            list: A list of dictionaries containing metric configurations for energy and force.
        """
        return [
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": 'energy',
                "target": 'y',
            },
            {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": 'energy',
                "target": 'y',
            },
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": 'force',
                "target": 'dy',
            },
            {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": 'force',
                "target": 'dy',
            }
        ]

    def get_output(self, output_config=None):
        """
        Get the output module for the MD17 task.
        
        Args:
            output_config: Configuration for the output module.
            
        Returns:
            torch.nn.ModuleList: A list containing the output module.
        """
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
        """
        Get the evaluator for the MD17 task.
        
        Returns:
            None: No special evaluator is needed for this task.
        """
        return None

    def get_dataloader_map(self):
        """
        Get the dataloader map for the MD17 task.
        
        Returns:
            list: A list containing 'test' as the only dataloader to use.
        """
        return ['test']
