from typing import List

import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from gotennet import utils

log = utils.get_logger(__name__)

import torch


@utils.task_wrapper
def test(cfg: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """
    mm_prec = cfg.get("matmul_precision", "high")
    log.info(f"Running with {mm_prec} precision.")
    torch.set_float32_matmul_precision(mm_prec)

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    if cfg.get("checkpoint"):
        from gotennet.models.goten_model import GotenModel

        model = GotenModel.from_pretrained(cfg.checkpoint)
        label = model.label
        if cfg.get("label", -1) == -1 and label is not None:
            cfg.label = label
    else:
        model = None

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    cfg.label_str = str(cfg.label)
    cfg.name = cfg.label_str + "_" + cfg.name

    if type(cfg.label) == str and hasattr(datamodule, "dataset_class"):
        cfg.label = datamodule.dataset_class().label_to_idx(cfg.label)
        log.info(f"Label {cfg.label} is mapped to index {cfg.label}")

        datamodule.label = cfg.label

    dataset_meta = (
        datamodule.get_metadata(cfg.label)
        if hasattr(datamodule, "get_metadata")
        else None
    )

    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    if model is None:
        model: LightningModule = hydra.utils.instantiate(
            cfg.model, dataset_meta=dataset_meta
        )

    print(model)

    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for name, cb_conf in cfg.callbacks.items():
            if name not in ["model_summary", "rich_progress_bar"]:
                continue
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": cfg.ckpt_path})

    log.info("Starting testing!")

    if cfg.get("ckpt_path"):
        ckpt_path = cfg.ckpt_path
    else:
        ckpt_path = None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics
    metric_dict = test_metrics
    return metric_dict
