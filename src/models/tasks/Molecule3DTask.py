"""  """

from __future__ import print_function, absolute_import, division

import torch
import torch.nn.functional as F
import torchmetrics
from src.datamodules.components.molx import Molecule3D
from torch.nn import L1Loss

from src.models.components.outputs import Dipole, Atomwise, PassThrough
from src.models.tasks.Task import Task

MOLECULE_TASK_DEFAULTS = {
    "decentralize": False,
}


class Molecule3DTask(Task):
    name = "Molecule3D"

    def __init__(self, representation, label_key, dataset_meta, task_config=None, **kwargs):
        super().__init__(representation, label_key, dataset_meta, task_config=task_config,
                         task_defaults=MOLECULE_TASK_DEFAULTS, **kwargs)
        self.num_classes = 1

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
            return f"{Molecule3D.available_properties[metric_meta['target']]}"
        return super(Molecule3DTask, self).get_metric_names(metric_meta, metric_idx)

    def get_losses(self):
        if self.label_key == Molecule3D.muxyz:
            return [{
                "metric": F.mse_loss,
                "prediction": "property",
                "target": self.label_key,
                "loss_weight": 1.
            }]

        return [{
            "metric": L1Loss,
            "prediction": "property",
            "target": self.label_key,
            "loss_weight": 1.
        }]

    def get_metrics(self):
        if self.label_key == Molecule3D.muxyz:
            return [
                {
                    "metric": torchmetrics.MeanSquaredError,
                    "prediction": 'property',
                    "target": self.label_key,
                }
            ]

        return [
            {
                "metric": torchmetrics.MeanSquaredError,
                "prediction": "property",
                "target": self.label_key,
            }, {
                "metric": torchmetrics.MeanAbsoluteError,
                "prediction": "property",
                "target": self.label_key,
            }
        ]

    def get_output(self, output_config=None):
        label_name = Molecule3D.available_properties[self.label_key]

        if output_config.get("pass_through", False):
            return [
                PassThrough(n_in=self.representation.hidden_dim, n_hidden=output_config.n_hidden, property="property")]

        if label_name == Molecule3D.mu:  # QM93D.mu
            outputs = Dipole(n_in=self.representation.hidden_dim, n_hidden=output_config.n_hidden,
                             predict_magnitude=True,
                             property="property")
        elif label_name == Molecule3D.muxyz:  # QM93D.muxyz
            outputs = Dipole(n_in=self.representation.hidden_dim, n_hidden=output_config.n_hidden,
                             predict_magnitude=False, decentralize_pos=self.config["decentralize"],
                             property="property")
        else:

            outputs = Atomwise(
                n_in=self.representation.hidden_dim,
                mean=self.dataset_meta['mean'][self.label_key],
                stddev=self.dataset_meta['stddev'][self.label_key],
                atomref=self.dataset_meta['atomref'],
                property="property",
                activation=F.silu,
                n_hidden=output_config.n_hidden
            )
        outputs = [outputs]
        return torch.nn.ModuleList(outputs)

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ['test']
