"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from controlcharts.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_loads() -> None:
    config = Config.from_yaml(REPO_ROOT / "configs" / "default.yaml")

    assert config.training.epochs == 60
    assert config.training.seed == 101
    assert config.paths.dataset.endswith("synthetic_control.csv")
    assert set(config.specifications) == {"diameter", "width"}
    assert config.specifications["diameter"].usl == 40.1


def test_missing_sections_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("training:\n  epochs: 5\n")
    config = Config.from_yaml(path)

    assert config.training.epochs == 5
    assert config.training.seed == 101
    assert config.paths.model.endswith(".keras")
    assert config.specifications == {}


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert Config.from_yaml(path).training.epochs == 60


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("tuning:\n  lr: 0.1\n")
    with pytest.raises(ValueError, match="unknown config sections"):
        Config.from_yaml(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown TrainingConfig keys"):
        Config.from_yaml(path)
