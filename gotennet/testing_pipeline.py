import os
from typing import List

import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from gotennet import utils

log = utils.get_logger(__name__)


@utils.task_wrapper
def test(cfg: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(cfg.ckpt_path):
        cfg.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    cfg.label_str = str(cfg.label)
    cfg.name = cfg.label_str + "_" + cfg.name

    if type(cfg.label) == str and hasattr(datamodule, 'dataset_class'):
        cfg.label = datamodule.dataset_class().label_to_idx(cfg.label)
        log.info(f"Label {cfg.label} is mapped to index {cfg.label}")
    dataset_meta = datamodule.get_metadata(cfg.label) if hasattr(datamodule, 'get_metadata') else None

    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, dataset_meta=dataset_meta)

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": cfg.ckpt_path})

    log.info("Starting testing!")

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
