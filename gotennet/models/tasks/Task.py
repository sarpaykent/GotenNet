"""Base class for all tasks in the project."""

from __future__ import absolute_import, division, print_function

import torch

from gotennet.utils import get_logger

log = get_logger(__name__)

class Task:
    """
    Base class for all tasks in the project.
    
    This class defines the interface for all tasks and provides common functionality.
    """

    name = None

    def __init__(
        self,
        representation,
        label_key,
        dataset_meta,
        task_config=None,
        task_defaults=None,
        **kwargs
    ):
        """
        Initialize a task.
        
        Args:
            representation: The representation model to use.
            label_key: The key for the label in the dataset.
            dataset_meta: Metadata about the dataset.
            task_config (dict, optional): Configuration for the task. Defaults to None.
            task_defaults (dict, optional): Default configuration for the task. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
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

    def process_outputs(
        self,
        batch,
        result,
        metric_meta,
        metric_idx
    ):
        """
        Process the outputs of the model for metric computation.
        
        Args:
            batch: The batch of data.
            result: The result of the model.
            metric_meta: Metadata about the metric.
            metric_idx: Index of the metric.
            
        Returns:
            tuple: A tuple containing the processed predictions and targets.
        """
        pred = result[metric_meta["prediction"]]
        targets = batch[metric_meta["target"]]
        pred = pred.reshape(targets.shape)

        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)

        return pred, targets

    def get_metric_names(
        self,
        metric_meta,
        metric_idx=0
    ):
        """
        Get the names of the metrics.
        
        Args:
            metric_meta: Metadata about the metric.
            metric_idx (int, optional): Index of the metric. Defaults to 0.
            
        Returns:
            str: The name of the metric.
        """
        return f"{metric_meta['prediction']}"

    def get_losses(self):
        """
        Get the loss functions for the task.
        
        Returns:
            list: A list of dictionaries containing loss function configurations.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("get_losses() is not implemented")

    def get_metrics(self):
        """
        Get the metrics for the task.
        
        Returns:
            list: A list of dictionaries containing metric configurations.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("get_metrics() is not implemented")

    def get_output(self, output_config=None):
        """
        Get the output module for the task.
        
        Args:
            output_config: Configuration for the output module.
            
        Returns:
            torch.nn.Module: The output module.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("get_output() is not implemented")

    def get_evaluator(self):
        """
        Get the evaluator for the task.
        
        Returns:
            object: The evaluator for the task, or None if not needed.
        """
        return None

    def get_dataloader_map(self):
        """
        Get the dataloader map for the task.
        
        Returns:
            list: A list of dataloader names to use for the task.
        """
        return ['test']
