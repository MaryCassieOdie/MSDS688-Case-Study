"""Transfer-learning models: EfficientNet-B0, ResNet50 and MobileNetV2.

Each backbone is built with ``input_tensor=`` so its layers live directly in
the outer model. That keeps fine-tuning simple (unfreeze the top N layers)
and lets Grad-CAM address the last convolutional layer by name.

Training is two-phase:
  1. *Feature extraction* - backbone frozen, only the new head learns.
  2. *Fine-tuning* - the top ``fine_tune_layers`` backbone layers are
     unfrozen and trained with a 10x lower learning rate. BatchNorm layers
     stay frozen throughout, which is the standard recipe for small datasets.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import config


@dataclass(frozen=True)
class Backbone:
    name: str
    builder: str            # attribute name in keras.applications
    preprocess_module: str  # module holding preprocess_input
    last_conv_layer: str    # used by Grad-CAM
    fine_tune_layers: int   # how many top backbone layers to unfreeze


BACKBONES: dict[str, Backbone] = {
    "EfficientNetB0": Backbone("EfficientNetB0", "EfficientNetB0", "efficientnet", "top_activation", 40),
    "ResNet50": Backbone("ResNet50", "ResNet50", "resnet50", "conv5_block3_out", 40),
    "MobileNetV2": Backbone("MobileNetV2", "MobileNetV2", "mobilenet_v2", "out_relu", 40),
}


def preprocess_fn(backbone_name: str) -> Callable:
    import keras

    module = getattr(keras.applications, BACKBONES[backbone_name].preprocess_module)
    return module.preprocess_input


def build_model(
    backbone_name: str,
    num_classes: int | None = None,
    img_size=config.IMG_SIZE,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
):
    import keras

    spec = BACKBONES[backbone_name]
    num_classes = num_classes or len(config.CLASS_NAMES)
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    base = getattr(keras.applications, spec.builder)(
        include_top=False, weights=weights, input_tensor=inputs
    )
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name=backbone_name)
    set_backbone_trainable(model, n_top=0)
    return model


HEAD_LAYERS = {"gap", "head_dropout", "predictions"}


def backbone_layers(model) -> list:
    import keras

    return [l for l in model.layers
            if l.name not in HEAD_LAYERS and not isinstance(l, keras.layers.InputLayer)]


def set_backbone_trainable(model, n_top: int) -> None:
    """Freeze the backbone except its top ``n_top`` layers (BatchNorm always frozen)."""
    import keras

    layers = backbone_layers(model)
    for i, layer in enumerate(layers):
        trainable = n_top > 0 and i >= len(layers) - n_top
        if isinstance(layer, keras.layers.BatchNormalization):
            trainable = False
        layer.trainable = trainable


def compile_model(model, lr: float) -> None:
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def train_two_phase(
    backbone_name: str,
    train_ds,
    val_ds,
    class_weight: dict | None = None,
    head_epochs: int = 8,
    fine_tune_epochs: int = 12,
    head_lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    model_dir: Path = config.MODEL_DIR,
    weights: str | None = "imagenet",
    num_classes: int | None = None,
):
    """Train one backbone end-to-end and save the best checkpoint.

    Returns ``(model, history_dict, train_seconds)``.
    """
    import keras

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / f"{backbone_name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    model = build_model(backbone_name, num_classes=num_classes, weights=weights)
    t0 = time.time()

    compile_model(model, head_lr)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=class_weight, callbacks=callbacks, verbose=2)

    set_backbone_trainable(model, BACKBONES[backbone_name].fine_tune_layers)
    compile_model(model, fine_tune_lr)  # recompile so trainable changes take effect
    h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=len(h1.epoch),
                   epochs=len(h1.epoch) + fine_tune_epochs,
                   class_weight=class_weight, callbacks=callbacks, verbose=2)
    seconds = time.time() - t0

    history = {k: list(map(float, h1.history[k])) + list(map(float, h2.history.get(k, [])))
               for k in h1.history}
    history["fine_tune_start_epoch"] = len(h1.epoch)
    (model_dir / f"{backbone_name}_history.json").write_text(json.dumps(history))
    # Reload the best checkpoint across *both* phases (by val_accuracy).
    model = keras.models.load_model(ckpt)
    return model, history, seconds


def load_model(backbone_name: str, model_dir: Path = config.MODEL_DIR):
    import keras

    return keras.models.load_model(Path(model_dir) / f"{backbone_name}.keras")


def predict_dataset(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, probabilities)`` for a labelled dataset."""
    y_true, probs = [], []
    for x, y in ds:
        probs.append(model.predict_on_batch(x))
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def inference_ms_per_image(model, img_size=config.IMG_SIZE, batch: int = 32, repeats: int = 5) -> float:
    x = np.random.uniform(0, 255, size=(batch, *img_size, 3)).astype("float32")
    model.predict_on_batch(x)  # warm-up / graph trace
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_on_batch(x)
    return 1000 * (time.perf_counter() - t0) / (batch * repeats)
