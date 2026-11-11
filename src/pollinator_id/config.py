"""Project-wide configuration: species list, paths and training defaults.

The species list deliberately mixes IUCN-threatened bumble bees with common
look-alike pollinators. A model that tells a Critically Endangered
*Bombus affinis* apart from an abundant *Bombus impatiens* is far more useful
to reviewers than one that only separates bees from butterflies.

IUCN categories are *not* hard-coded here. They are looked up at run time
(see ``iucn.py``) so the project always reflects the current Red List.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Species:
    scientific_name: str
    common_name: str
    group: str  # coarse pollinator group, used for EDA and reporting


# Fine-grained set: 5 threatened bumble bees + common look-alikes + other pollinators.
SPECIES: list[Species] = [
    Species("Bombus affinis", "Rusty-patched Bumble Bee", "bumble bee"),
    Species("Bombus terricola", "Yellow-banded Bumble Bee", "bumble bee"),
    Species("Bombus pensylvanicus", "American Bumble Bee", "bumble bee"),
    Species("Bombus occidentalis", "Western Bumble Bee", "bumble bee"),
    Species("Bombus fervidus", "Golden Northern Bumble Bee", "bumble bee"),
    Species("Bombus impatiens", "Common Eastern Bumble Bee", "bumble bee"),
    Species("Bombus griseocollis", "Brown-belted Bumble Bee", "bumble bee"),
    Species("Bombus bimaculatus", "Two-spotted Bumble Bee", "bumble bee"),
    Species("Bombus vagans", "Half-black Bumble Bee", "bumble bee"),
    Species("Apis mellifera", "Western Honey Bee", "other bee"),
    Species("Xylocopa virginica", "Eastern Carpenter Bee", "other bee"),
    Species("Danaus plexippus", "Monarch", "butterfly"),
]

CLASS_NAMES: list[str] = [s.scientific_name for s in SPECIES]

# ------------------------------------------------------------------------ lite mode
# LITE mode is a fast, CPU-friendly configuration for testing the whole
# pipeline locally (~10 min end to end on a laptop CPU). It uses the same
# 12 species and the same code paths, just far less data, smaller images and
# fewer epochs. Results are NOT meaningful for the write-up - use full mode on
# a GPU for that. Enable with the environment variable POLLINATOR_LITE=1
# (the notebooks' first cell sets it for you via LITE_MODE).
LITE = os.environ.get("POLLINATOR_LITE", "0").strip().lower() in {"1", "true", "yes", "on"}

# --------------------------------------------------------------------------- paths
# PROJECT_DATA_DIR lets Colab keep data on Google Drive so downloads survive
# runtime resets; locally it defaults to <repo>/data. Lite mode uses separate
# folders so a quick test never mixes with (or overwrites) a full run.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: Path) -> Path:
    base = Path(os.environ.get(env_var, default))
    return base.with_name(base.name + "_lite") if LITE else base   # data -> data_lite


DATA_DIR = _dir("PROJECT_DATA_DIR", REPO_ROOT / "data")
IMAGE_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"
IUCN_CSV = DATA_DIR / "iucn_status.csv"
OCCURRENCES_CSV = DATA_DIR / "occurrences.csv"   # coordinates-only sample for the rule engine
OUTPUT_DIR = _dir("PROJECT_OUTPUT_DIR", REPO_ROOT / "outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# --------------------------------------------------------------------- data/training
#                     full mode           lite mode
IMAGES_PER_CLASS    = 30         if LITE else 400   # research-grade observations per species
IMG_SIZE            = (160, 160) if LITE else (224, 224)
BATCH_SIZE          = 16         if LITE else 32
HEAD_EPOCHS         = 3          if LITE else 8     # phase 1: frozen backbone
FINE_TUNE_EPOCHS    = 2          if LITE else 12    # phase 2: top layers unfrozen
OCC_YEARS_BACK      = 4          if LITE else 14    # occurrence sample covers this many years
OCC_PER_YEAR        = 50         if LITE else 200   # occurrence records per species per year
NEEDS_ID_SAMPLE     = 12         if LITE else 60    # unidentified observations to triage
GRADCAM_SAMPLE      = 20         if LITE else 120   # test images for the attention check

PHOTO_LICENSES = "cc0,cc-by,cc-by-nc"  # only openly licensed photos are used
PHOTO_SIZE = "medium"           # iNaturalist size: square(75) small(240) medium(500) large(1024)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42

# Polite client identification for iNaturalist (no personal data).
USER_AGENT = "MSDS688-pollinator-case-study/1.0 (educational project)"


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGE_DIR, OUTPUT_DIR, MODEL_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
