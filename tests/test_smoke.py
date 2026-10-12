"""End to end: train briefly, evaluate, write figures and drive the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from controlcharts import plots
from controlcharts.cli import main
from controlcharts.config import TrainingConfig
from controlcharts.data import Dataset
from controlcharts.evaluate import evaluate, format_report
from controlcharts.model import load_model
from controlcharts.train import TrainingResult, save_model, train


@pytest.fixture(scope="module")
def quick_run(tiny_dataset_module: Dataset) -> TrainingResult:
    """One epoch on a small sample, shared by the tests below."""
    return train(tiny_dataset_module, TrainingConfig(epochs=1, test_size=0.25, seed=0))


@pytest.fixture(scope="module")
def tiny_dataset_module() -> Dataset:
    from controlcharts.data import generate

    return generate(samples_per_class=8, seed=0)


def test_training_one_epoch_produces_history(quick_run: TrainingResult) -> None:
    assert quick_run.epochs_run == 1
    assert set(quick_run.history) >= {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert 0.0 <= quick_run.final_val_accuracy <= 1.0


def test_training_is_reproducible(tiny_dataset_module: Dataset) -> None:
    config = TrainingConfig(epochs=1, test_size=0.25, seed=0)
    first = train(tiny_dataset_module, config)
    second = train(tiny_dataset_module, config)
    assert first.history["loss"] == pytest.approx(second.history["loss"])


def test_evaluate_scores_the_split(quick_run: TrainingResult) -> None:
    evaluation = evaluate(quick_run.model, quick_run.split.x_test, quick_run.split.y_test)

    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.confusion.shape == (6, 6)
    assert evaluation.confusion.sum() == len(quick_run.split.x_test)
    assert set(evaluation.per_class_accuracy()) == set(evaluation.class_names)
    assert "accuracy" in format_report(evaluation)


def test_model_round_trips_through_disk(quick_run: TrainingResult, tmp_path: Path) -> None:
    path = save_model(quick_run.model, tmp_path / "m.keras")
    reloaded = load_model(str(path))

    batch = quick_run.split.x_test
    assert np.allclose(
        quick_run.model.predict(batch, verbose=0),
        reloaded.predict(batch, verbose=0),
        atol=1e-6,
    )


def test_figures_are_written(
    quick_run: TrainingResult, tiny_dataset_module: Dataset, tmp_path: Path
) -> None:
    evaluation = evaluate(quick_run.model, quick_run.split.x_test, quick_run.split.y_test)

    written = [
        plots.plot_example_patterns(tiny_dataset_module, tmp_path / "patterns.png"),
        plots.plot_training_curves(quick_run.history, tmp_path / "curves.png"),
        plots.plot_confusion_matrix(evaluation, tmp_path / "confusion.png"),
    ]
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_cli_generate_then_train_then_predict(tmp_path: Path, real_csv: Path) -> None:
    """The documented command sequence, on a small generated dataset."""
    dataset_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.keras"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n"
        "  epochs: 1\n"
        "  test_size: 0.25\n"
        "  seed: 0\n"
        f"paths:\n"
        f"  dataset: {dataset_path}\n"
        f"  model: {model_path}\n"
        f"  assets: {tmp_path / 'assets'}\n"
        "specifications:\n"
        "  diameter:\n"
        "    usl: 40.1\n"
        "    lsl: 39.9\n"
    )

    assert main(["generate", "--output", str(dataset_path), "--samples-per-class", "8"]) == 0
    assert dataset_path.exists()

    assert main(["train", "--config", str(config_path)]) == 0
    assert model_path.exists()

    assert main(["evaluate", "--config", str(config_path)]) == 0
    assert (tmp_path / "assets" / "confusion_matrix.png").exists()

    assert main(["predict", str(real_csv), "--config", str(config_path), "--spec", "diameter"]) == 0


def test_cli_predict_output_is_json(tmp_path: Path, real_csv: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  model: models/control_charts.keras\n  assets: {tmp_path}\n")
    main(["predict", str(real_csv), "--config", str(config_path), "--usl", "40.1", "--lsl", "39.9"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["subgroups"] == 60
    assert set(payload["patterns"]) == {"x-bar", "range"}
    assert "cp" in payload and "cpk" in payload


def test_unknown_spec_name_is_rejected(tmp_path: Path, real_csv: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  model: models/control_charts.keras\n")
    with pytest.raises(KeyError, match="no specification named"):
        main(["predict", str(real_csv), "--config", str(config_path), "--spec", "thickness"])
