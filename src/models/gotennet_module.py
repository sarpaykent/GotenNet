from typing import Dict, Optional, Callable, TypeVar, List, Union, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import RankedLogger
from .tasks import TASK_DICT
from ..data.components.qm9 import QM9
from ..utils.utils import get_function_name

BaseModuleType = TypeVar('BaseModelType', bound='nn.Module')

log = RankedLogger(__name__, rank_zero_only=True)


class GotenNetModule(pl.LightningModule):
    """
    GotenNetModule: A PyTorch Lightning module for the GotenNet model.
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
            chain_scheduler: Optional[Callable] = None,
            weight_decay: float = 0.01,
            ema_decay: float = 0.9,
            dataset_meta: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
            output: Optional[Dict] = None,
            scheduler: Optional[Callable] = None,
            save_predictions: Optional[bool] = None,
            input_contribution: float = 1,
            task_config: Optional[Dict] = None,
            lr_warmup_steps: int = 0,
            schedule_free: bool = False,
            use_ema: bool = False,
            **kwargs
    ):
        super().__init__()
        self.use_ema = use_ema
        self.schedule_free = schedule_free
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr
        self.chain_scheduler = chain_scheduler
        self.input_contribution = input_contribution
        self.save_predictions = save_predictions
        self.task = task
        self.label = label

        if self.task in TASK_DICT:
            self.task_handler = TASK_DICT[self.task](representation, label, dataset_meta, task_config=task_config)
            self.evaluator = self.task_handler.get_evaluator()
        else:
            self.task_handler = None
            self.evaluator = None

        self.train_meta: List = []
        self.train_metrics: List = []
        self.val_meta = self.get_metrics()
        self.val_metrics = nn.ModuleList([v['metric']() for v in self.val_meta])
        self.test_meta = self.get_metrics()
        self.test_metrics = nn.ModuleList([v['metric']() for v in self.test_meta])

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay
        self.dataset_meta = dataset_meta

        self.ema_loss: Optional[float] = None
        self.ema_decay = ema_decay
        self.scheduler = scheduler

        self.save_hyperparameters()

        self.representation = representation
        self.output_modules = self.get_output(output or {})

        self.loss_meta = self.get_losses()
        for loss in self.loss_meta:
            if "ema_rate" in loss:
                if "ema_stages" not in loss:
                    loss["ema_stages"] = ['train', 'validation']
        self.loss_metrics = self.get_losses()
        self.loss_modules = nn.ModuleList([l["metric"]() for l in self.get_losses()])

        self.ema: Dict[str, Optional[float]] = {}
        for loss in self.get_losses():
            for stage in ['train', 'validation', 'test']:
                self.ema[f'{stage}_{loss["target"]}'] = None

        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])

    def configure_model(self):
        """Configure the model."""
        pass

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Save additional information to the checkpoint."""
        checkpoint["ema_loss"] = self.ema_loss

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Load additional information from the checkpoint."""
        if "ema_loss" in checkpoint:
            print("Loading ema loss")
            self.ema_loss = checkpoint["ema_loss"]
        else:
            print("No ema loss found")

    def get_losses(self) -> List[Dict]:
        """Get the loss configurations."""
        if self.task_handler:
            return self.task_handler.get_losses()
        return []

    def get_metrics(self) -> List[Dict]:
        """Get the metric configurations."""
        if self.task_handler:
            return self.task_handler.get_metrics()
        raise NotImplementedError("Task is not implemented")

    def get_phase_metric(self, phase: str = 'train') -> Tuple[List, List]:
        """Get the metrics for a specific phase."""
        if phase == 'train':
            return self.train_meta, self.train_metrics
        elif phase == 'validation':
            return self.val_meta, self.val_metrics
        elif phase == 'test':
            return self.test_meta, self.test_metrics
        raise NotImplementedError()

    def get_output(self, output_config: Optional[Dict] = None) -> List[nn.Module]:
        """Get the output modules."""
        if self.task_handler:
            return self.task_handler.get_output(output_config)
        raise NotImplementedError("Task is not implemented")

    def _get_num_graphs(self, batch: Union[torch.Tensor, List, Tuple]) -> int:
        """Get the number of graphs in a batch."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.num_graphs

    def calculate_output(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate the output for a batch."""
        result = {}
        for output_model in self.output_modules:
            result.update(output_model(batch))
        return result

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform a training step."""
        self._enable_grads(batch)
        batch.base_properties = {}
        batch.representation, batch.vector_representation = self.representation(batch)
        result = self.calculate_output(batch)
        loss = self.calculate_loss(batch, result, name="train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Perform a validation step."""
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)
        result = self.calculate_output(batch)
        torch.set_grad_enabled(False)
        val_loss = self.calculate_loss(batch, result, "validation").detach().item()
        self.log_metrics(batch, result, "validation")
        losses = {'val_loss': val_loss}
        self.log("validation/val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=self._get_num_graphs(batch))
        if self.evaluator:
            eval_keys = self.task_handler.get_evaluation_keys()
            losses['outputs'] = {
                "y_pred": result[eval_keys['pred']].detach().cpu(),
                "y_true": batch[eval_keys['target']].detach().cpu()
            }
        return losses

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Perform a test step."""
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)
        result = self.calculate_output(batch)
        torch.set_grad_enabled(False)
        test_loss = self.calculate_loss(batch, result).detach().item()
        self.log_metrics(batch, result, "test")
        torch.set_grad_enabled(False)

        losses = {
            loss_dict["prediction"]: result[loss_dict["prediction"]].cpu() for loss_index, loss_dict in
            enumerate(self.loss_meta)
        }
        if self.evaluator:
            eval_keys = self.task_handler.get_evaluation_keys()
            losses['outputs'] = {
                "y_pred": result[eval_keys['pred']].detach().cpu(),
                "y_true": batch[eval_keys['target']].detach().cpu()
            }
        return losses

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode the input batch."""
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)
        return batch

    def forward(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        torch.set_grad_enabled(True)
        self._enable_grads(batch)
        batch.representation, batch.vector_representation = self.representation(batch)
        result = self.calculate_output(batch)
        torch.set_grad_enabled(False)
        return result

    def log_metrics(self, batch: torch.Tensor, result: Dict[str, torch.Tensor], mode: str) -> None:
        """Log metrics for a specific mode."""
        for idx, (metric_meta, metric_module) in enumerate(zip(*self.get_phase_metric(mode))):
            loss_fn = metric_module
            if "target" in metric_meta.keys():
                if self.evaluator:
                    pred, targets = self.evaluator.process_outputs(batch, result, metric_meta, idx)
                else:
                    pred = result[metric_meta["prediction"]]
                    if self.task == "QM9":
                        if batch.y.shape[1] == 1:
                            targets = batch.y
                        else:
                            targets = batch.y[:, metric_meta["target"]]
                    elif self.task == "M3D":
                        from src.data.components.molecule3d import Molecule3D
                        targets = batch.labels[:, metric_meta["target"]]
                        if metric_meta["target"] == Molecule3D.label_to_idx(Molecule3D.muxyz):
                            targets = batch.dipole_xyz
                    else:
                        targets = batch[metric_meta["target"]]
                    pred = pred.reshape(targets.shape)
                    targets = targets.type(torch.float64)
                    pred = pred.type(torch.float64)

                loss_i = loss_fn(pred[:, :], targets).detach().item() if metric_meta[
                                                                             "prediction"] == "force" else loss_fn(pred,
                                                                                                                   targets).detach().item()
            else:
                loss_i = loss_fn(result[metric_meta["prediction"]]).detach().item()

            lossname = get_function_name(loss_fn)
            var_name = self.evaluator.get_metric_name(metric_meta,
                                                      idx) if self.evaluator else f"{QM9.available_properties[metric_meta['target']]}" if self.task == "QM9" else f"{metric_meta['prediction']}"

            self.log(
                f"{mode}/{lossname}_{var_name}",
                loss_i,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_num_graphs(batch)
            )

    def calculate_loss(self, batch: torch.Tensor, result: Dict[str, torch.Tensor],
                       name: Optional[str] = None) -> torch.Tensor:
        """Calculate the loss for a batch."""
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.use_ema:
            og_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for loss_index, loss_dict in enumerate(self.loss_meta):
            loss_fn = self.loss_modules[loss_index]
            if "target" in loss_dict.keys():
                if self.evaluator:
                    pred, targets = self.evaluator.process_outputs(batch, result, loss_dict, loss_index)
                else:
                    pred = result[loss_dict["prediction"]]
                    if self.task == "QM9":
                        if batch.y.shape[1] == 1:
                            targets = batch.y
                        else:
                            targets = batch.y[:, loss_dict["target"]]
                    elif self.task == "M3D":
                        from src.data.components.molecule3d import Molecule3D
                        targets = batch.labels[:, loss_dict["target"]]
                        if loss_dict["target"] == Molecule3D.label_to_idx(Molecule3D.muxyz):
                            targets = batch.dipole_xyz
                    else:
                        targets = batch[loss_dict["target"]]
                    pred = pred.reshape(targets.shape)
                    targets = targets.type(torch.float64)
                    pred = pred.type(torch.float64)
                loss_i = loss_fn(pred, targets)
            else:
                loss_i = loss_fn(result[loss_dict["prediction"]])

            ema_addon = ''
            if self.use_ema:
                og_loss += loss_dict["loss_weight"] * loss_i
            if 'ema_rate' in loss_dict and name in loss_dict['ema_stages'] and (1.0 > loss_dict['ema_rate'] > 0.0):
                ema_key = f'{name}_{loss_dict["target"]}'
                ema_addon = '_ema'
                if self.ema[ema_key] is None:
                    self.ema[ema_key] = loss_i.detach()
                else:
                    loss_ema = loss_dict['ema_rate'] * loss_i + (1 - loss_dict['ema_rate']) * self.ema[ema_key]
                    self.ema[ema_key] = loss_ema.detach()
                    if self.use_ema:
                        loss_i = loss_ema

            if name:
                self.log(
                    f"{name}/{loss_dict['prediction']}{ema_addon}_loss",
                    loss_i,
                    on_step=name == "train",
                    on_epoch=True,
                    prog_bar=name == "train",
                    batch_size=self._get_num_graphs(batch)
                )
            loss += loss_dict["loss_weight"] * loss_i

        if self.use_ema:
            self.log(
                f"{name}/val_loss_og",
                og_loss,
                on_step=name == "train",
                on_epoch=True,
                batch_size=self._get_num_graphs(batch)
            )

        return loss

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        print("self.weight_decay", self.weight_decay)
        if self.schedule_free:
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFreeClosure(
                self.trainer.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=1e-8,
                warmup_steps=self.lr_warmup_steps
            )
            return [optimizer], []
        else:
            optimizer = opt.AdamW(
                self.trainer.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=1e-7,
            )

        if self.scheduler and callable(self.scheduler):
            scheduler, _ = self.scheduler(optimizer=optimizer)
            schedule = {
                "scheduler": scheduler,
                "monitor": self.lr_monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            }
        else:
            scheduler = ReduceLROnPlateau(
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
        """Perform an optimization step."""
        optimizer = kwargs.get("optimizer", args[2])
        if not self.schedule_free:
            if self.trainer.global_step < self.hparams.lr_warmup_steps:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def _enable_grads(self, batch: torch.Tensor) -> None:
        """Enable gradients for the batch if required."""
        if self.requires_dr:
            batch.pos.requires_grad_()
