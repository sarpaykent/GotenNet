"""Callbacks for the GotenNet model."""

from typing import Any, Dict, Optional

import lightning.pytorch as L
import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EMALossCallback(L.Callback):
    """
    Exponential Moving Average (EMA) Loss Callback.

    This callback calculates and logs the EMA of the validation loss.
    """

    def __init__(
            self,
            alpha: float = 0.99,
            soft_beta: float = 10,
            validation_loss_name: str = "val_loss",
            ema_log_name: str = "validation/ema_loss"
    ):
        """
        Initialize the EMALossCallback.

        Args:
            alpha (float): The decay factor for EMA calculation. Default is 0.99.
            soft_beta (float): The soft beta factor for loss capping. Default is 10.
            validation_loss_name (str): The name of the validation loss in the outputs. Default is "val_loss".
            ema_log_name (str): The name under which to log the EMA loss. Default is "validation/ema_loss".
        """
        super().__init__()
        self.alpha = alpha
        self.ema: Optional[torch.Tensor] = None
        self.num_batches: int = 0
        self.soft_beta = soft_beta
        self.total_loss: Optional[torch.Tensor] = None
        self.validation_loss_name = validation_loss_name
        self.ema_log_name = ema_log_name

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load from.
        """
        if "ema_loss" in state_dict:
            log.info("EMA loss loaded")
            self.ema = state_dict["ema_loss"]
        else:
            log.info("EMA loss not found in checkpoint")
            self.ema = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary.

        Returns:
            Dict[str, Any]: The state dictionary containing the EMA loss.
        """
        return {"ema_loss": self.ema}

    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """
        Called when the validation epoch begins.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        self.num_batches = 0
        self.total_loss = torch.tensor(0.0, device=pl_module.device)
        if self.ema is not None and isinstance(self.ema, torch.Tensor):
            self.ema = self.ema.to(pl_module.device)

    def on_validation_batch_end(
            self,
            trainer: "L.Trainer",
            pl_module: "L.LightningModule",
            outputs: Any,
            batch: Any,
            batch_idx: int,
            **kwargs
    ) -> None:
        """
        Called when a validation batch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
            outputs (Any): The outputs from the validation step.
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
            **kwargs: Additional keyword arguments.
        """
        self.total_loss += outputs[self.validation_loss_name]
        self.num_batches += 1

    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """
        Called when the validation epoch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        avg_loss = self.total_loss / self.num_batches
        if self.ema is None:
            self.ema = avg_loss
        else:
            if self.soft_beta is not None:
                avg_loss = torch.min(torch.stack([avg_loss, self.ema * self.soft_beta]))
            self.ema = self.alpha * self.ema + (1 - self.alpha) * avg_loss
        pl_module.log(self.ema_log_name, self.ema, on_step=False, on_epoch=True)
