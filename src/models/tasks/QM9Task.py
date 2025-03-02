"""  """

from __future__ import print_function, absolute_import, division

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import L1Loss

from src.data.components.qm9 import QM9
from src.models.components.outputs import Dipole, Atomwise, ElectronicSpatialExtentV2
from src.models.tasks.Task import Task


class QM9Task(Task):
    name = "QM9"

    def __init__(self, representation, label_key, dataset_meta, task_config=None, **kwargs):
        super().__init__(representation, label_key, dataset_meta, task_config, **kwargs)

        if type(label_key) == str:
            self.label_key = QM9.available_properties.index(label_key)
        self.num_classes = 1
        print("QM9Task: ", task_config)
        self.task_loss = self.task_config.get("task_loss", "L1Loss")
        self.output_module = self.task_config.get("output_module", None)

    def process_outputs(self, batch, result, metric_meta, metric_idx):
        pred = result[metric_meta["prediction"]]
        targets = batch.y[:, metric_meta["target"]]
        pred = pred.reshape(targets.shape)
        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)

        return pred, targets

    def get_metric_names(self, metric_meta, metric_idx=0):
        if metric_meta["prediction"] == "property":
            return f"{QM9.available_properties[metric_meta['target']]}"
        return super(QM9Task, self).get_metric_names(metric_meta, metric_idx)

    def get_losses(self):
        if self.task_loss == "L1Loss":
            return [{
                "metric": L1Loss,
                "prediction": "property",
                "target": self.label_key,
                "loss_weight": 1.
            }]
        elif self.task_loss == "MSELoss":
            return [{
                "metric": torch.nn.MSELoss,
                "prediction": "property",
                "target": self.label_key,
                "loss_weight": 1.
            }]

    def get_metrics(self):
        return [
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": "property",
                "target": self.label_key,
            }, {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": "property",
                "target": self.label_key,
            },
            # {
            #     "metric": torchmetrics.aggregation.CatMetric,
            #     "prediction": "property",
            #     "target": "saves2",
            #     "log": False,
            # }
        ]

    def get_output(self, output_config=None):
        label_name = QM9.available_properties[self.label_key]
        if label_name == QM9.mu:  # QM93D.mu
            mean = self.dataset_meta.get("mean", None)
            std = self.dataset_meta.get("std", None)
            outputs = Dipole(n_in=self.representation.hidden_dim,
                             predict_magnitude=True,
                             property="property",
                             mean=mean,
                             stddev=std,
                             **output_config)
        elif label_name == QM9.r2:  # QM93D.r2
            outputs = ElectronicSpatialExtentV2(n_in=self.representation.hidden_dim, property="property",
                                                **output_config)
        else:
            mean = self.dataset_meta.get("mean", None)
            std = self.dataset_meta.get("std", None)
            outputs = Atomwise(
                n_in=self.representation.hidden_dim,
                mean=mean,
                stddev=std,
                atomref=self.dataset_meta['atomref'],
                property="property",
                activation=F.silu,
                **output_config)
        outputs = [outputs]
        return torch.nn.ModuleList(outputs)

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ['test']
