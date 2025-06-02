# Standard library imports
from typing import Callable, Dict, Optional, TypeVar

# Related third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as opt
from omegaconf import DictConfig

from gotennet.utils import get_logger

from ..utils.utils import get_function_name

# Local application/library specific imports
from .tasks import TASK_DICT

BaseModuleType = TypeVar("BaseModelType", bound="nn.Module")

log = get_logger(__name__)


def lazy_instantiate(d):
    if isinstance(d, dict) or isinstance(d, DictConfig):
        for k, v in d.items():
            if k == "__target__":
                log.info(f"Lazy instantiation of {v} with hydra.utils.instantiate")
                d["_target_"] = d.pop("__target__")
            elif isinstance(v, dict) or isinstance(v, DictConfig):
                lazy_instantiate(v)
    return d


class GotenModel(pl.LightningModule):
    """
    Atomistic model for molecular property prediction.

    This model combines a representation module with task-specific output modules
    to predict molecular properties.
    """

    def __init__(
        self,
        label: int,
        representation: nn.Module,
        task: str = "QM9",
        lr: float = 5e-4,
        lr_decay: float = 0.5,
        lr_patience: int = 100,
        lr_minlr: float = 1e-6,
        lr_monitor: str = "validation/ema_val_loss",
        weight_decay: float = 0.01,
        cutoff: float = 5.0,
        dataset_meta: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
        output: Optional[Dict] = None,
        scheduler: Optional[Callable] = None,
        save_predictions: Optional[bool] = None,
        task_config: Optional[Dict] = None,
        lr_warmup_steps: int = 0,
        use_ema: bool = False,
        **kwargs,
    ):
        """
        Initialize the atomistic model.

        Args:
            label: Target property index to predict.
            representation: Neural network module for atom/molecule representation.
            task: Task name, must be in TASK_DICT. Default is "QM9".
            lr: Learning rate. Default is 5e-4.
            lr_decay: Learning rate decay factor. Default is 0.5.
            lr_patience: Patience for learning rate scheduler. Default is 100.
            lr_minlr: Minimum learning rate. Default is 1e-6.
            lr_monitor: Metric to monitor for LR scheduling. Default is "validation/ema_val_loss".
            weight_decay: Weight decay for optimizer. Default is 0.01.
            cutoff: Cutoff distance for interactions. Default is 5.0.
            dataset_meta: Dataset metadata. Default is None.
            output: Output module configuration. Default is None.
            scheduler: Learning rate scheduler. Default is None.
            save_predictions: Whether to save predictions. Default is None.
            task_config: Task-specific configuration. Default is None.
            lr_warmup_steps: Number of warmup steps for learning rate. Default is 0.
            use_ema: Whether to use exponential moving average. Default is False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.use_ema = use_ema
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr

        self.save_predictions = save_predictions
        if output is None:
            output = {}

        self.task = task
        self.label = label

        self.train_meta = []
        self.train_metrics = []

        self.cutoff = cutoff
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay
        self.dataset_meta = dataset_meta
        _dataset_obj = (
            dataset_meta.pop("dataset")
            if dataset_meta and "dataset" in dataset_meta
            else None
        )

        self.scheduler = scheduler

        self.save_hyperparameters()

        if isinstance(representation, DictConfig) and (
            "__target__" in representation or "_target_" in representation
        ):
            import hydra

            lazy_instantiate(representation)
            representation = hydra.utils.instantiate(representation)

        self.representation = representation

        if self.task in TASK_DICT:
            self.task_handler = TASK_DICT[self.task](
                representation, label, dataset_meta, task_config=task_config
            )
            self.evaluator = self.task_handler.get_evaluator()
        else:
            self.task_handler = None
            self.evaluator = None

        self.val_meta = self.get_metrics()
        self.val_metrics = nn.ModuleList([v["metric"]() for v in self.val_meta])
        self.test_meta = self.get_metrics()
        self.test_metrics = nn.ModuleList([v["metric"]() for v in self.test_meta])

        self.output_modules = self.get_output(output)

        self.loss_meta = self.get_losses()
        for loss in self.loss_meta:
            if "ema_rate" in loss:
                if "ema_stages" not in loss:
                    loss["ema_stages"] = ["train", "validation"]
        self.loss_metrics = self.get_losses()
        self.loss_modules = nn.ModuleList([l["metric"]() for l in self.get_losses()])

        self.ema = {}
        for loss in self.get_losses():
            for stage in ["train", "validation", "test"]:
                self.ema[f"{stage}_{loss['target']}"] = None

        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_url: str,  # Input is always a string
    ):
        from gotennet.utils.file import download_checkpoint

        ckpt_path = download_checkpoint(checkpoint_url)
        return cls.load_from_checkpoint(ckpt_path)

    def configure_model(self) -> None:
        """
        Configure the model. This method is called by PyTorch Lightning.
        """
        pass

    def get_losses(self) -> list:
        """
        Get loss functions for the model.

        Returns:
            list: List of loss function configurations.

        Raises:
            NotImplementedError: If task handler is not available.
        """
        if self.task_handler:
            return self.task_handler.get_losses()
        else:
            raise NotImplementedError()

    def get_metrics(self) -> list:
        """
        Get metrics for model evaluation.

        Returns:
            list: List of metric configurations.

        Raises:
            NotImplementedError: If task is not implemented.
        """
        if self.task_handler:
            return self.task_handler.get_metrics()
        else:
            raise NotImplementedError(f"Task not implemented {self.task}")

    def get_phase_metric(self, phase: str = "train") -> tuple:
        """
        Get metrics for a specific training phase.

        Args:
            phase: Training phase ('train', 'validation', or 'test'). Default is 'train'.

        Returns:
            tuple: Tuple of (metric_meta, metric_modules).

        Raises:
            NotImplementedError: If phase is not recognized.
        """
        if phase == "train":
            return self.train_meta, self.train_metrics
        elif phase == "validation":
            return self.val_meta, self.val_metrics
        elif phase == "test":
            return self.test_meta, self.test_metrics

        raise NotImplementedError()

    def get_output(self, output_config: Optional[Dict] = None) -> list:
        """
        Get output modules based on configuration.

        Args:
            output_config: Configuration for output modules. Default is None.

        Returns:
            list: List of output modules.

        Raises:
            NotImplementedError: If task is not implemented.
        """
        if self.task_handler:
            return self.task_handler.get_output(output_config)
        else:
            raise NotImplementedError("Task not implemented")

    def _get_num_graphs(self, batch) -> int:
        """
        Get the number of graphs in a batch.

        Args:
            batch: Batch of data.

        Returns:
            int: Number of graphs in the batch.
        """
        if type(batch) in [list, tuple]:
            batch = batch[0]

        return batch.num_graphs

    def calculate_output(self, batch) -> Dict:
        """
        Calculate model outputs for a batch.

        Args:
            batch: Batch of data.

        Returns:
            Dict: Dictionary of model outputs.
        """
        result = {}
        for output_model in self.output_modules:
            result.update(output_model(batch))
        return result

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        self._enable_grads(batch)

        batch.representation, batch.vector_representation = self.representation(batch)

        result = self.calculate_output(batch)
        loss = self.calculate_loss(batch, result, name="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Dict:
        """
        Perform a validation step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader. Default is 0.

        Returns:
            Dict: Dictionary of validation losses and outputs.
        """
        torch.set_grad_enabled(True)
        self._enable_grads(batch)

        batch.representation, batch.vector_representation = self.representation(batch)

        result = self.calculate_output(batch)

        torch.set_grad_enabled(False)
        val_loss = self.calculate_loss(batch, result, "validation").detach().item()
        self.log_metrics(batch, result, "validation")
        torch.set_grad_enabled(False)

        losses = {"val_loss": val_loss}
        self.log(
            "validation/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._get_num_graphs(batch),
        )
        if self.evaluator:
            eval_keys = self.task_handler.get_evaluation_keys()

            losses["outputs"] = {
                "y_pred": result[eval_keys["pred"]].detach().cpu(),
                "y_true": batch[eval_keys["target"]].detach().cpu(),
            }

        return losses

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Dict:
        """
        Perform a test step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader. Default is 0.

        Returns:
            Dict: Dictionary of test losses and outputs.
        """
        torch.set_grad_enabled(True)
        self._enable_grads(batch)

        batch.representation, batch.vector_representation = self.representation(batch)

        result = self.calculate_output(batch)

        torch.set_grad_enabled(False)

        _test_loss = self.calculate_loss(batch, result).detach().item()
        self.log_metrics(batch, result, "test")
        torch.set_grad_enabled(False)

        losses = {
            loss_dict["prediction"]: result[loss_dict["prediction"]].cpu()
            for loss_index, loss_dict in enumerate(self.loss_meta)
        }

        if self.evaluator:
            eval_keys = self.task_handler.get_evaluation_keys()

            losses["outputs"] = {
                "y_pred": result[eval_keys["pred"]].detach().cpu(),
                "y_true": batch[eval_keys["target"]].detach().cpu(),
            }

        return losses

    def encode(self, batch) -> object:
        """
        Encode a batch of data.

        Args:
            batch: Batch of data.

        Returns:
            batch: Batch with added representation.
        """
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)
        return batch

    def forward(self, batch) -> Dict:
        """
        Forward pass through the model.

        Args:
            batch: Batch of data.

        Returns:
            Dict: Model outputs.
        """
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)

        result = self.calculate_output(batch)
        torch.set_grad_enabled(False)
        return result

    def log_metrics(self, batch, result, mode: str) -> None:
        """
        Log metrics for a specific mode.

        Args:
            batch: Batch of data.
            result: Model outputs.
            mode: Mode ('train', 'validation', or 'test').
        """
        for idx, (metric_meta, metric_module) in enumerate(
            zip(*self.get_phase_metric(mode), strict=False)
        ):
            loss_fn = metric_module

            if "target" in metric_meta.keys():
                pred, targets = self.task_handler.process_outputs(
                    batch, result, metric_meta, idx
                )

                pred = pred[:, :] if metric_meta["prediction"] == "force" else pred
                loss_i = loss_fn(pred, targets).detach().item()
            else:
                loss_i = loss_fn(result[metric_meta["prediction"]]).detach().item()

            lossname = get_function_name(loss_fn)

            if self.task_handler:
                var_name = self.task_handler.get_metric_names(metric_meta, idx)

            self.log(
                f"{mode}/{lossname}_{var_name}",
                loss_i,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_num_graphs(batch),
            )

    def calculate_loss(self, batch, result, name: Optional[str] = None) -> torch.Tensor:
        """
        Calculate loss for a batch.

        Args:
            batch: Batch of data.
            result: Model outputs.
            name: Name of the phase ('train', 'validation', or 'test'). Default is None.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.use_ema:
            og_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for loss_index, loss_dict in enumerate(self.loss_meta):
            loss_fn = self.loss_modules[loss_index]

            if "target" in loss_dict.keys():
                pred, targets = self.task_handler.process_outputs(
                    batch, result, loss_dict, loss_index
                )
                loss_i = loss_fn(pred, targets)
            else:
                loss_i = loss_fn(result[loss_dict["prediction"]])

            ema_addon = ""
            if self.use_ema:
                og_loss += loss_dict["loss_weight"] * loss_i

            # Check if EMA should be calculated
            if (
                "ema_rate" in loss_dict
                and name in loss_dict["ema_stages"]
                and (1.0 > loss_dict["ema_rate"] > 0.0)
            ):
                # Calculate EMA loss
                ema_key = f"{name}_{loss_dict['target']}"
                ema_addon = "_ema"
                if self.ema[ema_key] is None:
                    self.ema[ema_key] = loss_i.detach()
                else:
                    loss_ema = (
                        loss_dict["ema_rate"] * loss_i
                        + (1 - loss_dict["ema_rate"]) * self.ema[ema_key]
                    )
                    self.ema[ema_key] = loss_ema.detach()
                    if self.use_ema:
                        loss_i = loss_ema

            if name:
                self.log(
                    f"{name}/{loss_dict['prediction']}{ema_addon}_loss",
                    loss_i,
                    on_step=True if name == "train" else False,
                    on_epoch=True,
                    prog_bar=True if name == "train" else False,
                    batch_size=self._get_num_graphs(batch),
                )
            loss += loss_dict["loss_weight"] * loss_i

        if self.use_ema:
            self.log(
                f"{name}/val_loss_og",
                og_loss,
                on_step=True if name == "train" else False,
                on_epoch=True,
                batch_size=self._get_num_graphs(batch),
            )

        return loss

    def configure_optimizers(self) -> tuple:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            tuple: Tuple of (optimizers, schedulers).
        """
        optimizer = opt.AdamW(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            # amsgrad=True, # changed based on gemnet
            eps=1e-7,
        )

        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, **self.scheduler
            )
        else:
            scheduler = opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        schedule = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [schedule]

    def optimizer_step(self, *args, **kwargs) -> None:
        """
        Perform an optimizer step with learning rate warmup.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]

        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def _enable_grads(self, batch) -> None:
        """
        Enable gradients for position tensor if derivatives are required.

        Args:
            batch: Batch of data.
        """
        if self.requires_dr:
            batch.pos.requires_grad_()
