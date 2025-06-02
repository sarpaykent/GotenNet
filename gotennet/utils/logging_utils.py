from __future__ import absolute_import, division, print_function

import logging

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """
    Initialize multi-GPU-friendly python command line logger.

    Args:
        name: Name of the logger, defaults to the module name.

    Returns:
        logging.Logger: Logger instance with rank zero only decorators.
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    Control which config parts are saved by Lightning loggers.

    Additionally saves:
    - number of model parameters (total, trainable, non-trainable)

    Args:
        config: DictConfig containing the hydra config.
        model: Lightning model.
        trainer: Lightning trainer.
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
