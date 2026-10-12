"""Configuration loaded from YAML into dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """How the model is trained.

    Attributes:
        epochs: Maximum training epochs.
        batch_size: Samples per gradient step.
        test_size: Fraction of the dataset held out for testing.
        seed: Seed for the split, weight initialization and shuffling.
        early_stopping_patience: Epochs without validation loss improvement
            before training stops.
        verbose: Keras verbosity level.
    """

    epochs: int = 60
    batch_size: int = 32
    test_size: float = 0.10
    seed: int = 101
    early_stopping_patience: int = 3
    verbose: int = 0


@dataclass(frozen=True)
class PathsConfig:
    """Where inputs and outputs live.

    Attributes:
        dataset: The labelled CSV to train on.
        model: Where the trained model is written and read.
        assets: Directory for figures.
    """

    dataset: str = "data/synthetic_control.csv"
    model: str = "models/control_charts.keras"
    assets: str = "assets"


@dataclass(frozen=True)
class SpecificationConfig:
    """A measured characteristic and its drawing tolerance.

    Attributes:
        name: Characteristic name, used in figure titles.
        usl: Upper specification limit.
        lsl: Lower specification limit.
    """

    name: str
    usl: float
    lsl: float


@dataclass(frozen=True)
class Config:
    """The whole configuration.

    Attributes:
        training: Training hyperparameters.
        paths: Input and output locations.
        specifications: Known characteristics, keyed by name.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    specifications: dict[str, SpecificationConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config file.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed config, with defaults for anything absent.

        Raises:
            ValueError: If the file holds keys this version does not know.
        """
        data = yaml.safe_load(Path(path).read_text()) or {}

        unknown = set(data) - {"training", "paths", "specifications"}
        if unknown:
            raise ValueError(f"{path}: unknown config sections {sorted(unknown)}")

        specs = {
            name: SpecificationConfig(name=name, **values)
            for name, values in (data.get("specifications") or {}).items()
        }
        return cls(
            training=_build(TrainingConfig, data.get("training"), path),
            paths=_build(PathsConfig, data.get("paths"), path),
            specifications=specs,
        )


def _build(cls: type, values: dict | None, path: str | Path):
    """Build a config dataclass, rejecting keys it does not define."""
    values = values or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{path}: unknown {cls.__name__} keys {sorted(unknown)}")
    return cls(**values)
