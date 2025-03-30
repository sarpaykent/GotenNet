"""QM9 task implementation for quantum chemistry property prediction."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import L1Loss

from gotennet.datamodules.components.qm9 import QM9
from gotennet.models.components.outputs import (
    Atomwise,
    Dipole,
    ElectronicSpatialExtentV2,
)
from gotennet.models.tasks.Task import Task


class QM9Task(Task):
    """
    Task for QM9 quantum chemistry dataset.

    This task predicts various quantum chemistry properties for small molecules.
    """

    name = "QM9"

    def __init__(
        self,
        representation: torch.nn.Module,
        label_key: str | int,
        dataset_meta: dict,
        task_config: dict | None = None,
        **kwargs
    ):
        """
        Initialize the QM9 task.

        Args:
            representation (torch.nn.Module): The representation model to use.
            label_key (str | int): The key or index for the label in the dataset.
            dataset_meta (dict): Metadata about the dataset (e.g., mean, std, atomref).
            task_config (dict, optional): Configuration for the task. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            representation,
            label_key,
            dataset_meta,
            task_config,
            **kwargs
        )

        if isinstance(label_key, str):
            self.label_key = QM9.available_properties.index(label_key)
        self.num_classes = 1
        self.task_loss = self.task_config.get("task_loss", "L1Loss")
        self.output_module = self.task_config.get("output_module", None)

    def process_outputs(
        self,
        batch,
        result: dict,
        metric_meta: dict,
        metric_idx: int
    ):
        """
        Process the outputs of the model for metric computation.

        Args:
            batch: The batch of data, expected to have a 'y' attribute for targets.
            result (dict): The dictionary containing model outputs (predictions).
            metric_meta (dict): Metadata about the metric, including 'prediction' and 'target' keys.
            metric_idx (int): Index of the metric.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed predictions and targets.
        """
        pred = result[metric_meta["prediction"]]
        if batch.y.shape[1] == 1:
            targets = batch.y
        else:
            targets = batch.y[:, metric_meta["target"]]
        pred = pred.reshape(targets.shape)
        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)

        return pred, targets

    def get_metric_names(
        self,
        metric_meta: dict,
        metric_idx: int = 0
    ):
        """
        Get the names of the metrics.

        Args:
            metric_meta (dict): Metadata about the metric.
            metric_idx (int, optional): Index of the metric. Defaults to 0.

        Returns:
            str: The name of the metric, potentially including the property name.
        """
        if metric_meta["prediction"] == "property":
            return f"{QM9.available_properties[metric_meta['target']]}"
        return super(QM9Task, self).get_metric_names(metric_meta, metric_idx)

    def get_losses(self) -> list[dict]:
        """
        Get the loss functions for the QM9 task.

        Returns:
            list[dict]: A list of dictionaries, each containing loss function configuration.
        """
        if self.task_loss == "L1Loss":
            return [
                {
                    "metric": L1Loss,
                    "prediction": "property",
                    "target": self.label_key,
                    "loss_weight": 1.
                }
            ]
        elif self.task_loss == "MSELoss":
            return [
                {
                    "metric": torch.nn.MSELoss,
                    "prediction": "property",
                    "target": self.label_key,
                    "loss_weight": 1.
                }
            ]

    def get_metrics(self) -> list[dict]:
        """
        Get the metrics for the QM9 task.

        Returns:
            list[dict]: A list of dictionaries, each containing metric configuration.
        """
        return [
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": "property",
                "target": self.label_key,
            },
            {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": "property",
                "target": self.label_key,
            },
        ]

    def get_output(self, output_config: dict | None = None) -> torch.nn.ModuleList:
        """
        Get the output module for the QM9 task based on the target property.

        Args:
            output_config (dict | None): Configuration for the output module.

        Returns:
            torch.nn.ModuleList: A list containing the appropriate output module.
        """
        label_name = QM9.available_properties[self.label_key]
        output_config = output_config or {} # Ensure output_config is a dict

        if label_name == QM9.mu:
            mean = self.dataset_meta.get("mean", None)
            std = self.dataset_meta.get("std", None)
            outputs = Dipole(
                n_in=self.representation.hidden_dim,
                predict_magnitude=True,
                property="property",
                mean=mean,
                stddev=std,
                **output_config,
            )
        elif label_name == QM9.r2:
            outputs = ElectronicSpatialExtentV2(
                n_in=self.representation.hidden_dim,
                property="property",
                **output_config,
            )
        else:
            # Default to Atomwise for other properties
            mean = self.dataset_meta.get("mean", None)
            std = self.dataset_meta.get("std", None)
            outputs = Atomwise(
                n_in=self.representation.hidden_dim,
                mean=mean,
                stddev=std,
                atomref=self.dataset_meta.get('atomref'), # Use .get for safety
                property="property",
                activation=F.silu,
                **output_config,
            )
        return torch.nn.ModuleList([outputs])

    def get_evaluator(self) -> None:
        """
        Get the evaluator for the QM9 task.

        Returns:
            None: No special evaluator is needed for this task.
        """
        return None

    def get_dataloader_map(self) -> list[str]:
        """
        Get the dataloader map for the QM9 task.

        Returns:
            list[str]: A list containing 'test' as the only dataloader phase to use.
        """
        return ['test']
