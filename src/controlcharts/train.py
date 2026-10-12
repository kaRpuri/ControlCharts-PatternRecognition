"""Training, with every source of randomness seeded."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np

from .config import TrainingConfig
from .data import Dataset
from .model import build_model
from .preprocess import Split, prepare

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, numpy and Keras so a run is reproducible.

    The original notebook shuffled with no seed, so its split and therefore
    its reported accuracy cannot be recovered.

    Args:
        seed: The seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


@dataclass
class TrainingResult:
    """A finished training run.

    Attributes:
        model: The trained model.
        split: The data it was trained and tested on.
        history: Per epoch metrics, keyed by metric name.
        epochs_run: Epochs actually completed before early stopping.
    """

    model: keras.Model
    split: Split
    history: dict[str, list[float]]
    epochs_run: int

    @property
    def final_accuracy(self) -> float:
        """Training accuracy at the last epoch."""
        return float(self.history["accuracy"][-1])

    @property
    def final_val_accuracy(self) -> float:
        """Validation accuracy at the last epoch."""
        return float(self.history["val_accuracy"][-1])


def train(dataset: Dataset, config: TrainingConfig | None = None) -> TrainingResult:
    """Train the classifier on a labelled dataset.

    The test split is used as the validation set, matching the notebook, so
    the reported validation accuracy is test accuracy.

    Args:
        dataset: The labelled series to train on.
        config: Training hyperparameters, or None for the defaults.

    Returns:
        The trained model with its split and history.
    """
    config = config or TrainingConfig()
    set_seed(config.seed)

    split = prepare(dataset, test_size=config.test_size, seed=config.seed)
    model = build_model(series_length=split.num_features)

    logger.info(
        "training on %d samples, validating on %d, up to %d epochs",
        len(split.x_train),
        len(split.x_test),
        config.epochs,
    )

    history = model.fit(
        split.x_train,
        split.y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(split.x_test, split.y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
            )
        ],
        verbose=config.verbose,
        shuffle=True,
    )

    result = TrainingResult(
        model=model,
        split=split,
        history={k: [float(v) for v in vs] for k, vs in history.history.items()},
        epochs_run=len(history.history["loss"]),
    )
    logger.info(
        "stopped after %d epochs, validation accuracy %.4f",
        result.epochs_run,
        result.final_val_accuracy,
    )
    return result


def save_model(model: keras.Model, path: str | Path) -> Path:
    """Write a trained model to disk.

    Args:
        model: The model to save.
        path: Destination, with a .keras extension.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("wrote model to %s", path)
    return path
