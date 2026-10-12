"""The network builds, runs a forward pass and matches the original layout."""

from __future__ import annotations

import numpy as np

from controlcharts.data import NUM_CLASSES, SERIES_LENGTH
from controlcharts.model import HIDDEN_UNITS, INPUT_UNITS, build_model

# The model saved from the original notebook has this many parameters.
# 60*60+60 input, 60*39+39 hidden, 39*6+6 output.
ORIGINAL_PARAM_COUNT = 6279


def test_builds_with_the_original_architecture() -> None:
    model = build_model()
    kinds = [layer.__class__.__name__ for layer in model.layers]

    assert kinds == ["Dense", "Dropout", "Dense", "Dropout", "Dense"]
    assert model.layers[0].units == INPUT_UNITS
    assert model.layers[2].units == HIDDEN_UNITS
    assert model.layers[4].units == NUM_CLASSES
    assert model.count_params() == ORIGINAL_PARAM_COUNT


def test_activations_match_the_notebook() -> None:
    """A linear input layer, a ReLU hidden layer and a softmax output."""
    model = build_model()
    assert model.layers[0].activation.__name__ == "linear"
    assert model.layers[2].activation.__name__ == "relu"
    assert model.layers[4].activation.__name__ == "softmax"


def test_forward_pass_returns_a_probability_distribution() -> None:
    model = build_model()
    batch = np.random.default_rng(0).uniform(0, 1, size=(4, SERIES_LENGTH))

    probabilities = model.predict(batch, verbose=0)

    assert probabilities.shape == (4, NUM_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(probabilities >= 0)


def test_is_compiled_for_training() -> None:
    model = build_model()
    assert model.loss == "categorical_crossentropy"
    assert model.optimizer is not None


def test_accepts_a_single_sample() -> None:
    model = build_model()
    probabilities = model.predict(np.zeros((1, SERIES_LENGTH)), verbose=0)
    assert probabilities.shape == (1, NUM_CLASSES)
