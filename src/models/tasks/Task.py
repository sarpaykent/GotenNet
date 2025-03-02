"""  """

from __future__ import print_function, absolute_import, division

import torch

from src.utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)


class Task:
    name = None

    def __init__(self, representation, label_key, dataset_meta, task_config=None, task_defaults=None, **kwargs):
        if task_config is None:
            task_config = {}
        if task_defaults is None:
            task_defaults = {}

        self.task_config = task_config

        self.config = {**task_defaults, **task_config}
        log.info(f"Task config: {self.config}")
        self.representation = representation
        self.label_key = label_key
        self.dataset_meta = dataset_meta

        self.cast_to_float64 = True

    def process_outputs(self, batch, result, metric_meta, metric_idx):
        pred = result[metric_meta["prediction"]]
        targets = batch[metric_meta["target"]]
        pred = pred.reshape(targets.shape)

        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)

        return pred, targets

    def get_metric_names(self, metric_meta, metric_idx=0):
        return f"{metric_meta['prediction']}"

    def get_losses(self):
        raise NotImplementedError("get_losses() is not implemented")

    def get_metrics(self):
        raise NotImplementedError("get_metrics() is not implemented")

    def get_output(self, output_config=None):
        raise NotImplementedError("get_output() is not implemented")

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ['test']
