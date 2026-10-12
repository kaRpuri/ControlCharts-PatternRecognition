"""The classifier network.

The architecture is the one from ModelBuilding.ipynb, unchanged: a linear
input layer of 60 units, dropout, a 39 unit ReLU hidden layer, a second
dropout that the notebook set to zero, and a 6 way softmax. The zero rate
dropout is a no-op and is kept so the layer count and the loaded weight
layout still match the model that was originally trained.
"""

from __future__ import annotations

import keras
from keras import layers

from .data import NUM_CLASSES, SERIES_LENGTH

INPUT_UNITS = 60
HIDDEN_UNITS = 39
INPUT_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.0


def build_model(
    series_length: int = SERIES_LENGTH,
    num_classes: int = NUM_CLASSES,
    input_units: int = INPUT_UNITS,
    hidden_units: int = HIDDEN_UNITS,
    input_dropout: float = INPUT_DROPOUT,
    hidden_dropout: float = HIDDEN_DROPOUT,
) -> keras.Model:
    """Build and compile the classifier.

    Args:
        series_length: Points per input series.
        num_classes: Number of output classes.
        input_units: Units in the linear input layer.
        hidden_units: Units in the ReLU hidden layer.
        input_dropout: Dropout rate after the input layer.
        hidden_dropout: Dropout rate after the hidden layer.

    Returns:
        A compiled model, ready to fit.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(series_length,)),
            layers.Dense(input_units),
            layers.Dropout(input_dropout),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dropout(hidden_dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def load_model(path: str) -> keras.Model:
    """Load a saved model.

    Args:
        path: Path to a .keras or legacy .h5 file.

    Returns:
        The loaded model, not compiled, which is all inference needs.
    """
    return keras.models.load_model(path, compile=False)
