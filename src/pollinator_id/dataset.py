"""Train/val/test splitting and ``tf.data`` input pipelines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def assign_splits(df: pd.DataFrame, fractions=config.SPLIT_FRACTIONS, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified split by species. One photo per observation means no
    individual animal appears in more than one split."""
    train_f, val_f, test_f = fractions
    train, rest = train_test_split(df, train_size=train_f, stratify=df["label"], random_state=seed)
    val, test = train_test_split(
        rest, train_size=val_f / (val_f + test_f), stratify=rest["label"], random_state=seed
    )
    out = df.copy()
    out.loc[train.index, "split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out


def class_weights(df: pd.DataFrame, class_names=config.CLASS_NAMES) -> dict[int, float]:
    """Inverse-frequency weights so rarer (often threatened) species are not
    drowned out by abundant ones."""
    counts = df[df["split"] == "train"]["label"].value_counts()
    total, k = counts.sum(), len(class_names)
    return {i: float(total / (k * counts.get(name, 1))) for i, name in enumerate(class_names)}


# ---------------------------------------------------------------- tf.data
def _augmenter():
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08, fill_mode="reflect"),
            keras.layers.RandomZoom((-0.15, 0.0), fill_mode="reflect"),
            keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
            keras.layers.RandomContrast(0.15),
        ],
        name="augment",
    )


def load_image(path, img_size=config.IMG_SIZE):
    import tensorflow as tf

    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, img_size)  # float32 in [0, 255]
    return img


def make_dataset(
    df: pd.DataFrame,
    split: str,
    preprocess_fn,
    image_dir: Path = config.IMAGE_DIR,
    class_names=config.CLASS_NAMES,
    batch_size: int = config.BATCH_SIZE,
    img_size=config.IMG_SIZE,
    augment: bool | None = None,
):
    """Build a batched dataset of (preprocessed_image, label_index).

    ``preprocess_fn`` is the backbone-specific ``keras.applications.*.preprocess_input``.
    Augmentation runs on raw [0, 255] pixels *before* preprocessing, so the
    same pipeline works for every backbone.
    """
    import tensorflow as tf

    part = df[df["split"] == split]
    index = {name: i for i, name in enumerate(class_names)}
    paths = [str(Path(image_dir) / fp) for fp in part["filepath"]]
    labels = part["label"].map(index).to_numpy(dtype=np.int32)

    augment = (split == "train") if augment is None else augment
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = _augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
