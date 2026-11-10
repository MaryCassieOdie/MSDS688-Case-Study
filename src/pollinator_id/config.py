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

# --------------------------------------------------------------------------- paths
# PROJECT_DATA_DIR lets Colab keep data on Google Drive so downloads survive
# runtime resets; locally it defaults to <repo>/data.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PROJECT_DATA_DIR", REPO_ROOT / "data"))
IMAGE_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"
IUCN_CSV = DATA_DIR / "iucn_status.csv"
OCCURRENCES_CSV = DATA_DIR / "occurrences.csv"   # coordinates-only sample for the rule engine
OUTPUT_DIR = Path(os.environ.get("PROJECT_OUTPUT_DIR", REPO_ROOT / "outputs"))
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# --------------------------------------------------------------------- data/training
IMAGES_PER_CLASS = 400          # research-grade observations downloaded per species
PHOTO_LICENSES = "cc0,cc-by,cc-by-nc"  # only openly licensed photos are used
PHOTO_SIZE = "medium"           # iNaturalist size: square(75) small(240) medium(500) large(1024)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42

# Polite client identification for iNaturalist (no personal data).
USER_AGENT = "MSDS688-pollinator-case-study/1.0 (educational project)"


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGE_DIR, OUTPUT_DIR, MODEL_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
