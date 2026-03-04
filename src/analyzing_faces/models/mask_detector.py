from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


@dataclass(slots=True)
class MaskPrediction:
    label: str
    probability: float


class MaskDetector:
    def __init__(self, model_path: Path) -> None:
        self.model = self._load_or_build(model_path)

    def _load_or_build(self, model_path: Path) -> tf.keras.Model:
        if model_path.exists():
            return tf.keras.models.load_model(model_path)

        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def predict(self, bgr_face_crop: np.ndarray) -> MaskPrediction:
        resized = cv2.resize(bgr_face_crop, (128, 128))
        arr = np.expand_dims(resized, axis=0)
        prob_mask = float(self.model.predict(arr, verbose=0)[0][0])

        if prob_mask >= 0.5:
            return MaskPrediction(label="MASK", probability=prob_mask)
        return MaskPrediction(label="NO_MASK", probability=1.0 - prob_mask)
