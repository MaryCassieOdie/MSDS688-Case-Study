"""Generate the project notebooks from plain-Python cell definitions.

Run ``python notebooks/_build_notebooks.py`` after editing this file. Keeping
the source here makes notebook diffs reviewable; the .ipynb files are the
artifacts you open in Colab/Jupyter.
"""
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).parent


def nb(cells):
    out = []
    for kind, src in cells:
        src = dedent(src).strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": lines})
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = ("code", '''
# --- Environment setup: works on Google Colab and locally -------------------------------
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/MaryCassieOdie/MSDS688-Case-Study.git"

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = "/content/MSDS688-Case-Study"
    if not os.path.exists(REPO_DIR):
        # If the repo is private, upload the project folder to Drive instead and set
        # REPO_DIR = "/content/drive/MyDrive/MSDS688-Case-Study"
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=False)
    # Data and outputs live on Drive so they survive Colab runtime resets.
    DRIVE_ROOT = "/content/drive/MyDrive/MSDS688-Case-Study"
    os.environ["PROJECT_DATA_DIR"] = f"{DRIVE_ROOT}/data"
    os.environ["PROJECT_OUTPUT_DIR"] = f"{DRIVE_ROOT}/outputs"
else:
    REPO_DIR = os.path.abspath("..")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
# Optional: official IUCN Red List API token (otherwise iNaturalist's IUCN mirror is used)
# os.environ["IUCN_API_TOKEN"] = "..."   # on Colab, prefer: from google.colab import userdata

from pollinator_id import config
config.ensure_dirs()
print("data   ->", config.DATA_DIR)
print("output ->", config.OUTPUT_DIR)
''')

LOCAL_IMAGES = ("code", '''
# Reading thousands of small files from Google Drive is slow; copy them to the
# Colab VM's local disk once per session for fast training.
import shutil
from pathlib import Path
IMAGE_DIR = config.IMAGE_DIR
if IN_COLAB:
    local = Path("/content/images")
    if not local.exists():
        shutil.copytree(config.IMAGE_DIR, local)
    IMAGE_DIR = local
print("images from", IMAGE_DIR)
''')

# =========================================================================== 01
NB1 = [
    ("md", '''
# 01 - Data collection & exploratory analysis
**Endangered Pollinator Identification System** - MSDS688 case study

This notebook builds the dataset behind the system:

1. **IUCN cross-reference table** - Red List category for every species in scope.
2. **Training images** - research-grade, openly licensed iNaturalist photos (one per observation).
3. **Occurrence sample** - year-stratified coordinates used later by the conservation rule engine.
4. **EDA** - class balance, IUCN mix, geography, time coverage.

### Why these species?
The scope mixes **five IUCN-threatened bumble bees** with **visually similar, common species**
(plus honey bee, carpenter bee and monarch). Telling a Critically Endangered *Bombus affinis*
from an abundant *Bombus impatiens* is the hard, useful problem; most unidentified iNaturalist
bee observations are stuck at genus *Bombus* for exactly this reason.
'''),
    SETUP,
    ("code", '''
import pandas as pd
import matplotlib.pyplot as plt
from pollinator_id import inaturalist, iucn, dataset
from pollinator_id.config import SPECIES, CLASS_NAMES

pd.DataFrame([s.__dict__ for s in SPECIES])
'''),
    ("md", '''
## 1. IUCN Red List cross-reference
Categories are fetched live, not hard-coded. With `IUCN_API_TOKEN` set, the official IUCN Red List
API v4 is used; otherwise the global IUCN assessment mirrored on each iNaturalist taxon record.
'''),
    ("code", '''
client = inaturalist.INatClient()
iucn_table = iucn.lookup_statuses(CLASS_NAMES, client)
iucn_table.to_csv(config.IUCN_CSV, index=False)
iucn_table[["scientific_name", "common_name", "iucn_code", "iucn_category", "threatened", "source"]]
'''),
    ("md", '''
## 2. Training images
Up to `IMAGES_PER_CLASS` of the most recent research-grade observations per species
(CC0 / CC-BY / CC-BY-NC photos, wild organisms only). Only the **first photo** of each
observation is used so the same individual cannot appear in both train and test.
Re-running is safe: existing metadata and images are reused.

*Expect ~2 minutes of API calls plus a few minutes of image downloads.*
'''),
    ("code", '''
if config.METADATA_CSV.exists():
    meta = pd.read_csv(config.METADATA_CSV)
    print(f"Loaded existing metadata: {len(meta)} rows")
else:
    frames = []
    for name in CLASS_NAMES:
        df = inaturalist.collect_species_metadata(client, name, n_images=config.IMAGES_PER_CLASS)
        print(f"{name:24s} {len(df):4d} observations")
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("observation_id")
    meta = inaturalist.download_images(meta)
    meta = dataset.assign_splits(meta)
    meta.to_csv(config.METADATA_CSV, index=False)

# Photo credits, required by the CC licences.
meta[["filepath", "url", "license", "attribution"]].to_csv(config.DATA_DIR / "ATTRIBUTION.csv", index=False)
meta.groupby(["label", "split"]).size().unstack()[["train", "val", "test"]]
'''),
    ("md", '''
## 3. Occurrence sample for the conservation rule engine
Up to 200 research-grade records **per species per year** (coordinates only, no images).
Stratifying by year keeps abundant species from being represented only by recent records,
which the "possible local decline" rule needs. Common species are included on purpose:
they act as a proxy for **survey effort** in each area.

*~13 years x 12 species = ~160 API calls, about 3-4 minutes.*
'''),
    ("code", '''
from datetime import date
YEARS = range(2012, date.today().year + 1)
if config.OCCURRENCES_CSV.exists():
    occ = pd.read_csv(config.OCCURRENCES_CSV)
else:
    occ = pd.concat([inaturalist.collect_occurrences(client, n, YEARS, per_year=200) for n in CLASS_NAMES],
                    ignore_index=True)
    occ.to_csv(config.OCCURRENCES_CSV, index=False)
print(len(occ), "occurrence records;", f"{occ.coordinates_obscured.mean():.1%} with obscured coordinates")
occ.pivot_table(index="label", columns="year", values="observation_id", aggfunc="count", fill_value=0)
'''),
    ("md", "## 4. Exploratory analysis"),
    ("code", '''
STATUS_COLORS = {"CR": "#b2182b", "EN": "#ef8a62", "VU": "#f4a582", "NT": "#fddbc7",
                 "LC": "#92c5de", "DD": "#bababa", "NE": "#e0e0e0"}
codes = iucn_table.set_index("scientific_name")["iucn_code"]

counts = meta["label"].value_counts().reindex(CLASS_NAMES)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(counts.index, counts.values, color=[STATUS_COLORS[codes[n]] for n in counts.index], edgecolor="k", lw=.4)
for i, n in enumerate(counts.index):
    ax.text(counts[n] + 3, i, codes[n], va="center", fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("images"); ax.set_title("Images per species (colour = IUCN category)")
fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "class_counts.png", dpi=150)
'''),
    ("code", '''
fig, ax = plt.subplots(figsize=(11, 6))
for name in CLASS_NAMES:
    sub = meta[meta.label == name]
    thr = codes[name] in iucn.THREATENED
    ax.scatter(sub.longitude, sub.latitude, s=10 if thr else 3, alpha=.8 if thr else .25,
               color=STATUS_COLORS[codes[name]] if thr else "gray",
               label=f"{name} [{codes[name]}]" if thr else None, zorder=3 if thr else 1)
ax.set_xlim(-170, -50); ax.set_ylim(15, 70)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=.3)
ax.set_title("Training observations - threatened species highlighted, others in gray")
ax.legend(fontsize=8, loc="lower left")
fig.savefig(config.FIGURE_DIR / "training_map.png", dpi=150, bbox_inches="tight")
'''),
    ("code", '''
meta["year"] = pd.to_datetime(meta.observed_on, errors="coerce").dt.year
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
meta.year.value_counts().sort_index().plot.bar(ax=axes[0], color="#4d9221")
axes[0].set_title("Training images by observation year")
(meta.groupby("label").coordinates_obscured.mean().reindex(CLASS_NAMES) * 100).plot.barh(ax=axes[1], color="#8073ac")
axes[1].set_title("% of observations with obscured coordinates"); axes[1].invert_yaxis()
fig.tight_layout()
'''),
    ("code", '''
# Sample images - note how similar several bumble bees look.
from PIL import Image
fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(10, 2.1 * len(CLASS_NAMES)))
for i, name in enumerate(CLASS_NAMES):
    for j, fp in enumerate(meta[meta.label == name].sample(5, random_state=1).filepath):
        axes[i, j].imshow(Image.open(config.IMAGE_DIR / fp)); axes[i, j].axis("off")
    axes[i, 0].set_title(f"{name} [{codes[name]}]", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "sample_images.png", dpi=110)
'''),
    ("md", '''
### Observations to discuss
* **Class balance** - are threatened species (e.g. *B. occidentalis*) under-represented? Class
  weights in notebook 02 compensate for any imbalance.
* **Geographic bias** - observations cluster near population centres; the rule engine treats
  sparse areas as *under-surveyed* rather than *empty*.
* **Obscured coordinates** - threatened taxa are often auto-obscured, which is why notebook 04
  uses a coarse spatial grid.
'''),
]

# =========================================================================== 02
NB2 = [
    ("md", '''
# 02 - Model comparison: EfficientNet-B0 vs ResNet50 vs MobileNetV2
All three ImageNet-pretrained backbones are trained with the **same data, splits, augmentation,
class weights and two-phase schedule**:

| Phase | Backbone | Learning rate | Epochs (max) |
|---|---|---|---|
| 1. Feature extraction | frozen | 1e-3 | 8 |
| 2. Fine-tuning | top 40 layers unfrozen (BatchNorm frozen) | 1e-4 | 12 |

Early stopping on validation accuracy, then evaluation on the held-out **test** split.
Besides accuracy we report **macro-F1** (every species counts equally), **top-3 accuracy**
(the system *recommends* candidates for expert review), and **threatened recall**.

> Runtime: ~10-15 min per model on a Colab T4 GPU. *Runtime -> Change runtime type -> T4 GPU*.
'''),
    SETUP,
    LOCAL_IMAGES,
    ("code", '''
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from pollinator_id import dataset, models, evaluation
from pollinator_id.config import CLASS_NAMES

print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
tf.keras.utils.set_random_seed(config.SEED)

meta = pd.read_csv(config.METADATA_CSV)
iucn_table = pd.read_csv(config.IUCN_CSV)
iucn_codes = dict(zip(iucn_table.scientific_name, iucn_table.iucn_code))
cw = dataset.class_weights(meta)
print(meta.split.value_counts().to_dict())
print({CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()})
'''),
    ("code", '''
BACKBONES = ["EfficientNetB0", "ResNet50", "MobileNetV2"]
RETRAIN = False  # set True to retrain even if a saved model exists

results, histories, test_probs = [], {}, {}
for name in BACKBONES:
    print(f"\\n========== {name} ==========")
    prep = models.preprocess_fn(name)
    train_ds = dataset.make_dataset(meta, "train", prep, image_dir=IMAGE_DIR)
    val_ds = dataset.make_dataset(meta, "val", prep, image_dir=IMAGE_DIR)
    test_ds = dataset.make_dataset(meta, "test", prep, image_dir=IMAGE_DIR)

    ckpt = config.MODEL_DIR / f"{name}.keras"
    hist_file = config.MODEL_DIR / f"{name}_history.json"
    time_file = config.MODEL_DIR / f"{name}_train_seconds.txt"
    if ckpt.exists() and hist_file.exists() and not RETRAIN:
        model = models.load_model(name)
        history = json.loads(hist_file.read_text())
        seconds = float(time_file.read_text()) if time_file.exists() else float("nan")
        print("loaded saved model")
    else:
        model, history, seconds = models.train_two_phase(name, train_ds, val_ds, class_weight=cw)
        time_file.write_text(str(seconds))

    y_true, probs = models.predict_dataset(model, test_ds)
    ms = models.inference_ms_per_image(model)
    results.append(evaluation.summarize(name, y_true, probs, model.count_params(), seconds, ms,
                                        iucn_codes=iucn_codes))
    histories[name], test_probs[name] = history, (y_true, probs)
    np.savez(config.MODEL_DIR / f"{name}_test_predictions.npz", y_true=y_true, probs=probs)

res = pd.DataFrame(results).set_index("model")
res.to_csv(config.OUTPUT_DIR / "model_comparison.csv")
res.style.format("{:.3f}").highlight_max(subset=["accuracy", "top3_accuracy", "macro_f1", "threatened_recall"],
                                         color="#c7e9c0")
'''),
    ("md", "## Learning curves"),
    ("code", '''
fig = evaluation.plot_histories(histories)
fig.savefig(config.FIGURE_DIR / "learning_curves.png", dpi=150)
'''),
    ("md", "## Accuracy vs. cost"),
    ("code", '''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, r in res.iterrows():
    axes[0].scatter(r.params_millions, r.macro_f1, s=120); axes[0].annotate(name, (r.params_millions, r.macro_f1), xytext=(6, 4), textcoords="offset points")
    axes[1].scatter(r.ms_per_image, r.macro_f1, s=120); axes[1].annotate(name, (r.ms_per_image, r.macro_f1), xytext=(6, 4), textcoords="offset points")
axes[0].set_xlabel("parameters (millions)"); axes[1].set_xlabel("inference ms / image (batch 32)")
for a in axes: a.set_ylabel("test macro-F1"); a.grid(alpha=.3)
fig.suptitle("Model size and speed vs. accuracy"); fig.tight_layout()
fig.savefig(config.FIGURE_DIR / "accuracy_vs_cost.png", dpi=150)
'''),
    ("md", "## Confusion matrices (row-normalised, test split)"),
    ("code", '''
fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
for ax, name in zip(axes, BACKBONES):
    evaluation.plot_confusion(*test_probs[name], title=name, ax=ax)
fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "confusion_matrices.png", dpi=130)
'''),
    ("md", "## Per-species F1 - where do the models differ?"),
    ("code", '''
f1 = pd.DataFrame({n: evaluation.per_class_report(*test_probs[n]).loc[CLASS_NAMES, "f1-score"] for n in BACKBONES})
ax = f1.plot.barh(figsize=(10, 7), width=.8)
ax.set_yticklabels([f"{n} [{iucn_codes[n]}]" for n in f1.index]); ax.invert_yaxis()
ax.set_xlabel("F1 (test)"); ax.set_title("Per-species F1 by backbone (IUCN category in brackets)"); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(config.FIGURE_DIR / "per_species_f1.png", dpi=150)
f1.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None)
'''),
    ("md", '''
## Select the model used downstream
The deployed system **recommends** species to human experts, and the costliest error is missing a
threatened species. We therefore pick the model with the best **macro-F1**, breaking ties by
**threatened recall**. Change `SELECTION_METRIC` to choose differently (e.g. `ms_per_image` for a
mobile/field deployment, where MobileNetV2 is designed to shine).
'''),
    ("code", '''
SELECTION_METRIC = "macro_f1"
best = res.sort_values([SELECTION_METRIC, "threatened_recall"], ascending=False).index[0]
(config.OUTPUT_DIR / "best_model.json").write_text(json.dumps({"backbone": best, "selected_by": SELECTION_METRIC}))
print("Selected backbone:", best)
evaluation.per_class_report(*test_probs[best]).round(3)
'''),
    ("md", '''
### Discussion prompts
* Which bumble-bee pairs are most confused (confusion matrices)? Are they colour-pattern look-alikes,
  e.g. *B. affinis* vs. *B. griseocollis* / *B. impatiens*?
* Does the larger ResNet50 actually beat the far smaller EfficientNet-B0 / MobileNetV2?
* How large is the gap between top-1 and top-3 accuracy? A big gap supports a
  "shortlist for expert review" product rather than an automatic ID.
* Compare `threatened_flag_recall` (right *status*, even if wrong species) with `threatened_recall`.
'''),
]

# =========================================================================== 03
NB3 = [
    ("md", '''
# 03 - Explainability with Grad-CAM
Grad-CAM (Selvaraju et al., 2017) highlights the image regions that most increased the
predicted class score. For this system it answers two questions:

1. **Is the model using the right evidence?** Identification of bumble bees depends on the
   colour banding of the thorax and abdomen. A model that looks at the *flower* or *background*
   is exploiting dataset shortcuts (e.g. a species photographed mostly on one plant).
2. **Can an expert trust a suggestion?** Showing the heatmap next to the suggested ID lets
   reviewers quickly see what the model "looked at".
'''),
    SETUP,
    ("code", '''
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pollinator_id import dataset, models, gradcam
from pollinator_id.config import CLASS_NAMES

meta = pd.read_csv(config.METADATA_CSV)
test = meta[meta.split == "test"].reset_index(drop=True)
iucn_codes = dict(pd.read_csv(config.IUCN_CSV)[["scientific_name", "iucn_code"]].values)
best = json.loads((config.OUTPUT_DIR / "best_model.json").read_text())["backbone"]
loaded = {n: models.load_model(n) for n in models.BACKBONES if (config.MODEL_DIR / f"{n}.keras").exists()}
print("best:", best, "| available:", list(loaded))

def raw(fp):
    return dataset.load_image(str(config.IMAGE_DIR / fp)).numpy()
'''),
    ("md", "## Correct predictions on threatened species (best model)"),
    ("code", '''
preds = np.load(config.MODEL_DIR / f"{best}_test_predictions.npz")
test["pred"] = [CLASS_NAMES[i] for i in preds["probs"].argmax(1)]
test["conf"] = preds["probs"].max(1)
# make_dataset keeps the order of the test rows (no shuffle outside "train"), so rows align.
assert (test.label.map(CLASS_NAMES.index).to_numpy() == preds["y_true"]).all()

threatened = [n for n in CLASS_NAMES if iucn_codes[n] in {"CR", "EN", "VU"}]
picks = (test[(test.label == test.pred) & test.label.isin(threatened)]
         .sort_values("conf", ascending=False).drop_duplicates("label"))
fig = gradcam.plot_gallery(loaded[best], best, [raw(f) for f in picks.filepath], picks.label.tolist(),
                           save_path=config.FIGURE_DIR / f"gradcam_threatened_{best}.png")
'''),
    ("md", "## Misclassifications - what went wrong?"),
    ("code", '''
wrong = test[test.label != test.pred].sort_values("conf", ascending=False).head(6)
print(wrong[["label", "pred", "conf"]].to_string())
if len(wrong):
    fig = gradcam.plot_gallery(loaded[best], best, [raw(f) for f in wrong.filepath], wrong.label.tolist(),
                               save_path=config.FIGURE_DIR / f"gradcam_errors_{best}.png")
'''),
    ("md", "## Same images, three backbones"),
    ("code", '''
sample = test.groupby("label").sample(1, random_state=7).head(5)
imgs = [raw(f) for f in sample.filepath]
fig, axes = plt.subplots(len(loaded) + 1, len(imgs), figsize=(2.7 * len(imgs), 2.8 * (len(loaded) + 1)))
for j, (img, true) in enumerate(zip(imgs, sample.label)):
    axes[0, j].imshow(img.astype("uint8")); axes[0, j].set_title(true, fontsize=8)
    for i, (name, m) in enumerate(loaded.items(), start=1):
        ov, _, idx, p = gradcam.explain(m, name, img)
        axes[i, j].imshow(ov)
        axes[i, j].set_title(f"{name}: {CLASS_NAMES[idx].split()[1]} {p[idx]:.0%}", fontsize=8,
                             color="tab:green" if CLASS_NAMES[idx] == true else "tab:red")
for a in axes.ravel(): a.axis("off")
fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "gradcam_backbone_comparison.png", dpi=130)
'''),
    ("md", '''
## A simple quantitative check: is attention on the insect?
Without bounding boxes we cannot measure this exactly, but iNaturalist photographers usually centre
the subject. We compute the share of Grad-CAM mass inside the central 50% of the image for every
test image. A model relying on background context will score noticeably lower.
'''),
    ("code", '''
def center_share(hm):
    h, w = hm.shape; t = hm.sum() + 1e-8
    return hm[h // 4: 3 * h // 4, w // 4: 3 * w // 4].sum() / t

rows = []
for name, m in loaded.items():
    for fp in test.sample(min(120, len(test)), random_state=0).filepath:
        _, hm, _, _ = gradcam.explain(m, name, raw(fp))
        rows.append({"model": name, "center_share": center_share(hm)})
cs = pd.DataFrame(rows)
ax = cs.boxplot(column="center_share", by="model", figsize=(7, 4)); plt.suptitle("")
ax.axhline(0.25, ls="--", c="gray"); ax.set_title("Grad-CAM mass in central 50% of image (dashed = uniform)")
cs.groupby("model").center_share.describe().round(3)
'''),
    ("md", '''
### Discussion prompts
* Do heatmaps concentrate on the abdomen/thorax banding that field guides use?
* For errors: did the model look at the right body part but confuse similar patterns, or did it
  look at the flower/background (a data problem)?
* Grad-CAM uses the last conv layer (7x7 for all three backbones at 224 px), so maps are coarse.
'''),
]

# =========================================================================== 04
NB4 = [
    ("md", '''
# 04 - Identification workflow & conservation recommendations
This notebook runs the complete system from the architecture slide:

**iNaturalist image -> CNN -> predicted species -> IUCN Red List status -> recommendation**

* **Part A - Needs-ID triage.** Fetch recent iNaturalist bumble-bee observations that are still
  identified only to genus, suggest a species shortlist, cross-reference the Red List, and put
  potential threatened species at the top of the expert-review queue.
* **Part B - Where to invest.** A transparent rule-based system ranks geographic grid cells for
  conservation investment, and answers "where should we act to protect species *X*?".
'''),
    SETUP,
    ("code", '''
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pollinator_id import inaturalist, iucn, models, inference, gradcam
from pollinator_id import conservation_rules as cr
from pollinator_id.config import CLASS_NAMES

best = json.loads((config.OUTPUT_DIR / "best_model.json").read_text())["backbone"]
model = models.load_model(best)
iucn_table = pd.read_csv(config.IUCN_CSV)
redlist = iucn.RedListReference(iucn_table)
occ = pd.read_csv(config.OCCURRENCES_CSV)
client = inaturalist.INatClient()
print("model:", best)
'''),
    ("md", '''
## Part A - Triage of unidentified observations
**Important caveat:** the classifier is *closed-set*: it can only answer with one of the 12 species
it was trained on, while North America has ~50 bumble-bee species. Low-confidence suggestions are
labelled *uncertain*, and every output is a **recommendation for expert review**, never an
automatic identification.
'''),
    ("code", '''
needs = inference.fetch_needs_id_observations(client, "Bombus", n=60)
queue = inference.triage(model, best, needs, redlist, review_threshold=0.5)
cols = ["url", "place_guess", "observed_on", "pred_label", "confidence", "pred_iucn_code", "review_note", "alternatives"]
queue[cols].head(20).style.format({"confidence": "{:.0%}"})
'''),
    ("code", '''
show = queue.head(8).reset_index(drop=True)
fig, axes = plt.subplots(2, len(show), figsize=(2.7 * len(show), 7), gridspec_kw={"hspace": 0.35})
for j, r in show.iterrows():
    ov, *_ = gradcam.explain(model, best, r.image)
    axes[0, j].imshow(r.image.astype("uint8"))
    axes[0, j].set_title(f"obs {r.observation_id}", fontsize=8)
    axes[1, j].imshow(ov)
    axes[1, j].set_title(f"{r.pred_label.split()[1]} [{r.pred_iucn_code}]\\n{r.confidence:.0%} - {r.review_note}",
                         fontsize=8, color="tab:red" if r.pred_threatened else "black")
for a in axes.ravel(): a.axis("off")
fig.suptitle("Top of the expert-review queue (red = suggested threatened species)")
fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "triage_queue.png", dpi=130)
'''),
    ("code", '''
summary = queue.groupby(["pred_label", "pred_iucn_code"]).agg(n=("observation_id", "size"),
                                                               mean_conf=("confidence", "mean"))
print(f"{queue.pred_threatened.sum()} of {len(queue)} unidentified observations suggested as threatened species")
summary.sort_values("n", ascending=False).round(2)
'''),
    ("md", '''
## Part B - Rule-based conservation recommendations
Occurrence records are binned into a 1 deg x 1 deg grid. Each rule that fires in a cell adds
priority points and a plain-language recommendation:

| Rule | Points | Fires when |
|---|---|---|
| Critical refuge | 5 | CR/EN species with >= 3 records in the last 5 years |
| Vulnerable stronghold | 3 | VU species with >= 3 recent records |
| Threatened co-occurrence hotspot | 3 | >= 2 threatened species recently present |
| Possible local decline | 2 | threatened species with >= 3 older records but none recent |
| Under-surveyed area | 1 | threatened species present but < 20 records of any study species |
| Pending expert review | 1 | model flagged an unverified threatened sighting (>= 70% confidence) |

Tiers: **High** >= 8, **Medium** 4-7, **Low** 1-3. Thresholds live in `RuleSettings` and every
ranking lists the rules that produced it, so planners can audit or adjust the logic.
'''),
    ("code", '''
settings = cr.RuleSettings(cell_deg=1.0, recent_years=5)
pending = queue[["latitude", "longitude", "pred_label", "pred_iucn_code", "confidence"]].dropna()
cells = cr.evaluate(occ, iucn_table, settings, pending=pending)
cells.to_csv(config.OUTPUT_DIR / "conservation_priorities.csv", index=False)
print(cells.priority_tier.value_counts().to_dict())
cells.head(15)[["cell_id", "center_lat", "center_lon", "priority_score", "priority_tier",
                "threatened_species", "rules_fired"]]
'''),
    ("code", '''
ax = cr.plot_priority_map(cells, occ, title="Conservation priority by 1 deg grid cell")
ax.set_xlim(-170, -50); ax.set_ylim(15, 70)
plt.savefig(config.FIGURE_DIR / "priority_map.png", dpi=150, bbox_inches="tight")
'''),
    ("code", '''
# Interactive map (click a cell for the rules that fired). folium is preinstalled on Colab.
try:
    fmap = cr.folium_map(cells, settings.cell_deg)
    fmap.save(str(config.FIGURE_DIR / "priority_map.html"))
    display(fmap)
except ImportError:
    print("pip install folium for the interactive map")
'''),
    ("md", "### Where should we invest to protect a given species?"),
    ("code", '''
TARGET = "Bombus affinis"   # try any threatened species in scope
rec = cr.recommend_for_species(TARGET, occ, iucn_table, settings, top_n=10)
print(f"Top locations for {TARGET} [{redlist.status(TARGET)['iucn_code']}]")
rec[["cell_id", "center_lat", "center_lon", "priority_tier", "species_recent", "species_total",
     "species_last_year", "rules_fired"]]
'''),
    ("code", '''
for _, r in rec.head(3).iterrows():
    print(f"\\n## Cell {r.cell_id} ({r.center_lat:.1f}, {r.center_lon:.1f}) - {r.priority_tier} priority")
    for line in r.recommendations.split(" | "):
        print("  -", line)
'''),
    ("md", '''
### Sensitivity analysis
Rules encode judgement calls. How stable are the rankings under different grid sizes and
definitions of "recent"? Rankings that persist across settings are the most defensible investments.
'''),
    ("code", '''
rows = []
for deg in (0.5, 1.0, 2.0):
    for yrs in (3, 5, 8):
        c = cr.evaluate(occ, iucn_table, cr.RuleSettings(cell_deg=deg, recent_years=yrs))
        rows.append({"cell_deg": deg, "recent_years": yrs, "cells_scored": len(c),
                     "high": int((c.priority_tier == "High").sum()),
                     "decline_flags": int(c.rules_fired.str.contains("decline").sum())})
pd.DataFrame(rows).pivot_table(index="cell_deg", columns="recent_years", values=["high", "decline_flags"])
'''),
    ("md", '''
## Limitations & responsible use
* **Closed-set classifier** - species outside the 12 classes will be forced into one of them;
  confidence thresholds and expert review are essential. Extending to all regional *Bombus*
  species is the natural next step.
* **Sampling bias** - citizen-science records follow people, roads and cities. "No recent
  records" can mean "nobody looked", which is why the under-surveyed rule recommends monitoring
  before investment.
* **Location sensitivity** - precise locations of threatened species can enable collection or
  disturbance. Outputs here use 1 deg cells and respect iNaturalist's coordinate obscuring;
  do not publish finer-grained threatened-species locations.
* **Status data** - IUCN categories are global; regional/national listings (e.g. US ESA,
  Canada SARA) may differ and could be added as additional rules.
* **Licensing** - photos are CC0/CC-BY/CC-BY-NC; credits are in `data/ATTRIBUTION.csv`. Use is
  non-commercial and educational.
'''),
]


if __name__ == "__main__":
    for fname, cells in [
        ("01_data_collection_eda.ipynb", NB1),
        ("02_model_comparison.ipynb", NB2),
        ("03_gradcam_explainability.ipynb", NB3),
        ("04_identification_and_conservation.ipynb", NB4),
    ]:
        (HERE / fname).write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", fname)
