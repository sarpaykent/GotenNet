"""
Utility functions for the GotenNet project.
"""

from __future__ import absolute_import, division, print_function

import os
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from gotennet.utils.logging_utils import get_logger

log = get_logger(__name__)


def find_config_directory() -> str:
    """
    Find the configs directory by searching in multiple locations.

    Returns:
        str: Absolute path to the configs directory.

    Raises:
        FileNotFoundError: If configs directory is not found in any search location.
    """
    package_location = os.path.dirname(
        os.path.realpath(__file__)
    )  # This will be utils.py's location
    current_dir = os.getcwd()

    # Define search paths in order of preference
    search_paths = [
        os.path.join(current_dir, "configs"),  # Check for configs in CWD
        os.path.join(
            current_dir, "gotennet", "configs"
        ),  # Check for gotennet/configs in CWD (e.g. running from project root)
        os.path.abspath(
            os.path.join(package_location, "..", "configs")
        ),  # Check for ../configs relative to utils.py (i.e. gotennet/configs)
    ]

    # Search for configs directory
    for path in search_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Set PROJECT_ROOT environment variable based on current_dir
            # This assumes that if configs are found, current_dir is likely the project root.
            os.environ["PROJECT_ROOT"] = current_dir
            return os.path.abspath(path)

    # If no configs directory found, raise detailed error
    searched_paths_str = "\n".join(
        f"  - {p}" for p in search_paths
    )  # Renamed variable to avoid conflict
    raise FileNotFoundError(
        f"Could not find 'configs' directory in any of the following locations:\n"
        f"{searched_paths_str}\n\n"
        f"Please ensure the 'configs' directory exists in one of these locations.\n"
        f"Current working directory: {current_dir}\n"
        f"Package location (of this util.py file): {package_location}"
    )


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
