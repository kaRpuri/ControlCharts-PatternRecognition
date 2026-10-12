"""Shared fixtures. Everything here is small so the suite stays fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from controlcharts.data import Dataset, generate

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The repository's data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def shipped_dataset() -> Dataset:
    """The UCI synthetic control chart dataset shipped in data/."""
    from controlcharts.data import load_csv

    return load_csv(DATA_DIR / "synthetic_control.csv")


@pytest.fixture(scope="session")
def tiny_dataset() -> Dataset:
    """Six classes, eight series each, deterministic."""
    return generate(samples_per_class=8, seed=0)


@pytest.fixture(scope="session")
def real_csv() -> Path:
    """The diameter measurements from the production line."""
    return DATA_DIR / "diameter_real_process.csv"


@pytest.fixture
def subgroups() -> np.ndarray:
    """Sixty subgroups of four measurements around a mean of 10."""
    rng = np.random.default_rng(3)
    return 10.0 + rng.normal(0, 0.05, size=(60, 4))


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A four by sixty measurement CSV in the layout predict expects."""
    rng = np.random.default_rng(5)
    values = 25.0 + rng.normal(0, 0.1, size=(4, 60))
    path = tmp_path / "tiny.csv"
    header = ",".join(str(i) for i in range(60))
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in values)
    path.write_text(f"{header}\n{rows}\n")
    return path
