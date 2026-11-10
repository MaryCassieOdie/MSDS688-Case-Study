"""Gradient-weighted Class Activation Mapping (Grad-CAM).

Selvaraju et al., 2017 - https://arxiv.org/abs/1610.02391

For a target class c, Grad-CAM weights each feature map A^k of the last
convolutional layer by the spatially averaged gradient d y^c / d A^k, sums
them, and applies ReLU. The result is a coarse heatmap of the image regions
that most increased the class score - e.g. whether the model looked at the
bee's banding pattern or at the flower it was sitting on.
"""
from __future__ import annotations

import numpy as np

from . import config
from .models import BACKBONES, preprocess_fn


def gradcam_heatmap(model, image_batch: np.ndarray, conv_layer_name: str, class_index: int | None = None):
    """Return ``(heatmap[h, w] in [0, 1], class_index, probabilities)`` for one
    *preprocessed* image (batch of size 1)."""
    import keras
    import tensorflow as tf

    grad_model = keras.Model(model.inputs, [model.get_layer(conv_layer_name).output, model.output])
    x = tf.convert_to_tensor(image_batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))           # (channels,)
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)        # (h, w)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index, preds[0].numpy()


def overlay(raw_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0, 1] heatmap (jet colormap) onto a raw [0, 255] RGB image."""
    import matplotlib
    import tensorflow as tf

    h, w = raw_image.shape[:2]
    hm = tf.image.resize(heatmap[..., None], (h, w)).numpy()[..., 0]
    colored = matplotlib.colormaps["jet"](hm)[..., :3] * 255
    return np.clip((1 - alpha) * raw_image + alpha * colored, 0, 255).astype("uint8")


def explain(model, backbone_name: str, raw_image: np.ndarray, class_index: int | None = None):
    """Convenience wrapper: raw [0, 255] image -> (overlay, heatmap, class_index, probs)."""
    prep = preprocess_fn(backbone_name)
    x = prep(np.array(raw_image, dtype="float32", copy=True)[None, ...])
    heatmap, idx, probs = gradcam_heatmap(model, x, BACKBONES[backbone_name].last_conv_layer, class_index)
    return overlay(raw_image, heatmap), heatmap, idx, probs


def plot_gallery(model, backbone_name: str, raw_images, true_labels, class_names=config.CLASS_NAMES,
                 short_names: bool = True, save_path=None):
    """Grid: original image (top) and Grad-CAM overlay (bottom) per sample."""
    import matplotlib.pyplot as plt

    def short(name):
        g, s = name.split()[:2]
        return f"{g[0]}. {s}" if short_names else name

    n = len(raw_images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    axes = np.array(axes).reshape(2, n)
    for j, (img, true) in enumerate(zip(raw_images, true_labels)):
        ov, _, idx, probs = explain(model, backbone_name, img)
        pred = class_names[idx]
        ok = pred == true
        axes[0, j].imshow(img.astype("uint8"))
        axes[0, j].set_title(f"true: {short(true)}", fontsize=9)
        axes[1, j].imshow(ov)
        axes[1, j].set_title(f"pred: {short(pred)} ({probs[idx]:.0%})", fontsize=9,
                             color="tab:green" if ok else "tab:red")
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM - {backbone_name}", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
