import os
from typing import List, Tuple

import hydra
import torch.multiprocessing
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    seed_everything,
)

import gotennet.utils.logging_utils
from gotennet import utils

log = utils.get_logger(__name__)

import torch


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    mm_prec = cfg.get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(mm_prec)
    log.info(f"Running with {mm_prec} precision.")

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    ckpt_path = cfg.trainer.get("resume_from_checkpoint", None)

    # Convert relative ckpt path to absolute path if necessary
    if ckpt_path and not os.path.isabs(ckpt_path):
        cfg.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    cfg.label_str = str(cfg.label)
    cfg.name = cfg.label_str + "_" + cfg.name

    log.info(f"Label string is: {cfg.label_str}")

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

    model: LightningModule = hydra.utils.instantiate(
        cfg.model, dataset_meta=dataset_meta
    )

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for name, cb_conf in cfg.callbacks.items():
            if cfg.exp and name in ["learning_rate_monitor"]:
                continue
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in cfg and not cfg.exp:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    # profiler = PyTorchProfiler()
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
        inference_mode=False,
    )
    # trainer = Trainer(barebones=True)
    datamodule.device = model.device
    print("Current device is: ", model.device)
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    gotennet.utils.logging_utils.log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer,
    )

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Get metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )

    train_metrics = trainer.callback_metrics

    # Test the model
    if cfg.get("test"):
        # ckpt_path = "best"
        if cfg.get("train") and not cfg.trainer.get("fast_dev_run"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if not cfg.get("train") or cfg.trainer.get("fast_dev_run"):
            if cfg.get("ckpt_path"):
                ckpt_path = cfg.ckpt_path
            else:
                ckpt_path = None
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")

    # Print path to best checkpoint
    if not cfg.trainer.get("fast_dev_run") and cfg.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict
