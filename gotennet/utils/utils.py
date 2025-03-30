"""
Utility functions for the GotenNet project.
"""

from __future__ import absolute_import, division, print_function

from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from gotennet.utils import pylogger

log = pylogger.get_pylogger(__name__)

def task_wrapper(task_func: Callable) -> Callable:
    """
    Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)
    
    Args:
        task_func: The task function to wrap.
        
    Returns:
        Callable: The wrapped function.
        
    Example:
        ```
        @utils.task_wrapper
        def train(cfg: DictConfig) -> Tuple[dict, dict]:
            ...
            return metric_dict, object_dict
        ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float | None:
    """
    Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict (dict): Dictionary containing metrics logged by LightningModule.
        metric_name (str): Name of the metric to retrieve.

    Returns:
        float | None: The value of the metric, or None if metric_name is empty.

    Raises:
        Exception: If the metric name is provided but not found in the metric dictionary.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_function_name(func):
    if hasattr(func, "name"):
        func_name = func.name
    else:
        func_name = type(func).__name__.split(".")[-1]
    return func_name