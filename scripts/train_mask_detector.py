from pathlib import Path

import tensorflow as tf


# Minimal baseline trainer placeholder. Replace with production dataset pipeline.
def train_dummy_mask_detector(output_path: Path) -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


if __name__ == "__main__":
    train_dummy_mask_detector(Path("artifacts/mask_detector.keras"))
