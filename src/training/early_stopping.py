import logging
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This class monitors the validation loss and halts the training process if
    the loss does not decrease for a specified number of epochs. It can also
    restore the model's weights from the best-performing epoch.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        restore_best_weights: bool = True,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initializes the EarlyStopping instance.

        Args:
            patience: int, How many epochs to wait for a validation loss improvement before
                      stopping. Defaults to 5.
            delta: float, Minimum change in the monitored quantity to qualify as an improvement.
                   Defaults to 0.
            verbose: bool, If True, prints a message for each improvement and when early stopping
                     is triggered. Defaults to False.
            restore_best_weights: bool, If True, the model's weights from the best-performing
                                  epoch are restored upon early stopping. Defaults to True.
            logger: Optional[logging.Logger], Optional logger instance. If None and verbose is True, will print to console.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.logger = logger
        self.best_loss = None
        self.best_weights = None
        self.no_improvement_count = 0
        self.stop_training = False

    def _log(self, message: str, level: str = "info") -> None:
        """
        Internal method to handle logging/printing.

        Args:
            message: str, The message to log or print
            level: str, Logging level ('info', 'warning', 'debug'), default 'info'
        """
        if self.verbose:
            if self.logger:
                if level == "info":
                    self.logger.info(message)
                elif level == "warning":
                    self.logger.warning(message)
                elif level == "debug":
                    self.logger.debug(message)
            else:
                print(message)

    def check_early_stop(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Checks the validation loss and updates the internal state.

        Args:
            val_loss: float, The current validation loss.
            model: torch.nn.Module, The PyTorch model being trained.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            # Improvement detected
            improvement = self.best_loss - val_loss if self.best_loss is not None else 0
            self.best_loss = val_loss
            self.no_improvement_count = 0

            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

            if self.best_loss is not None and improvement > 0:
                self._log(
                    f"Validation loss improved by {improvement:.6f} to {val_loss:.6f}"
                )
            else:
                self._log(f"Initial validation loss: {val_loss:.6f}")
        else:
            # No improvement
            self.no_improvement_count += 1

            if self.no_improvement_count == 1:
                self._log(
                    f"No improvement in validation loss for {self.no_improvement_count} epoch",
                    level="debug",
                )
            else:
                self._log(
                    f"No improvement in validation loss for {self.no_improvement_count} epochs",
                    level="debug",
                )

            if self.no_improvement_count >= self.patience:
                self.stop_training = True

                self._log(
                    f"Early stopping triggered after {self.no_improvement_count} epochs without improvement",
                    level="warning",
                )

                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    self._log("Restored best model weights", level="info")

                self._log(f"Best validation loss: {self.best_loss:.6f}", level="info")
