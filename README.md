# Endangered Pollinator Identification System

MSDS688 case study, by Mary Odenthal.

A computer-vision system that identifies pollinator species from iNaturalist photos,
cross-references each prediction with the **IUCN Red List of Threatened Species**, and uses a
transparent **rule-based system** to recommend where conservation investment would help
threatened species most.

```
iNaturalist images --trains--> CNN (TensorFlow) --predicts--> species --references--> IUCN Red List
                                     |                                                    |
                                Grad-CAM heatmaps                    rule engine -> priority locations
```

## Why it matters
Many iNaturalist observations never get a species-level ID. Bumble bees are a clear case: thousands
of observations sit at genus *Bombus*, and threatened species such as the Critically Endangered
**rusty-patched bumble bee** (*Bombus affinis*) look like common relatives. Recommending a likely
species for **expert review**, and flagging possible threatened species first, speeds up
identification. Knowing where threatened species live and persist helps conservation spending
go where it has the most impact.

## Scope
The project covers 12 North American pollinators: five bumble bees that the IUCN lists as threatened,
plus common look-alikes and other pollinators.

| Species | Common name | Role in dataset |
|---|---|---|
| *Bombus affinis* | Rusty-patched Bumble Bee | threatened |
| *Bombus terricola* | Yellow-banded Bumble Bee | threatened |
| *Bombus pensylvanicus* | American Bumble Bee | threatened |
| *Bombus occidentalis* | Western Bumble Bee | threatened |
| *Bombus fervidus* | Golden Northern Bumble Bee | threatened |
| *Bombus impatiens*, *B. griseocollis*, *B. bimaculatus*, *B. vagans* | common bumble bees | look-alikes |
| *Apis mellifera*, *Xylocopa virginica*, *Danaus plexippus* | honey bee, carpenter bee, monarch | other pollinators |

The "threatened" column reflects the IUCN listings at the time of writing. The project never
hard-codes categories: notebook 01 looks them up live and saves them to `data/iucn_status.csv`.
To change the scope, edit `SPECIES` in `src/pollinator_id/config.py`.

## The case-study sub-topics
| Sub-topic | Where it lives |
|---|---|
| Comparison of EfficientNet-B0, ResNet50 and MobileNetV2 | `notebooks/02_model_comparison.ipynb`, `src/pollinator_id/models.py` |
| Focus on a category of endangered species (pollinators) | `src/pollinator_id/config.py` |
| Rule-based system recommending conservation locations | `notebooks/04_...ipynb`, `src/pollinator_id/conservation_rules.py` |
| Explainability with Grad-CAM | `notebooks/03_gradcam_explainability.ipynb`, `src/pollinator_id/gradcam.py` |

## Project layout
```
notebooks/
  01_data_collection_eda.ipynb          IUCN lookup, image download, occurrence sample, EDA
  02_model_comparison.ipynb             train + evaluate 3 backbones, select best
  03_gradcam_explainability.ipynb       heatmaps, error analysis, attention-on-subject check
  04_identification_and_conservation.ipynb   Needs-ID triage + rule-based recommendations
  _build_notebooks.py                   source the .ipynb files are generated from
scripts/
  run_lite.py            run notebooks 01-04 headlessly in lite (CPU test) mode
src/pollinator_id/
  config.py              species list, paths, hyperparameters, full/lite settings
  inaturalist.py         rate-limited iNaturalist API client and downloader
  iucn.py                IUCN Red List lookup (official API or iNaturalist mirror)
  dataset.py             stratified splits, class weights, tf.data pipelines
  models.py              backbone registry, two-phase transfer learning
  evaluation.py          metrics (macro-F1, top-3, threatened recall), plots
  gradcam.py             Grad-CAM heatmaps and galleries
  inference.py           image -> species -> IUCN status, Needs-ID triage queue
  conservation_rules.py  rule engine, species-specific recommendations, maps
data/      (created at run time, git-ignored) images, metadata, IUCN table, occurrences
outputs/   (created at run time, git-ignored) models, figures, result tables
```

## Running on Google Colab (recommended)
1. Push this repository to GitHub (the notebooks clone
   `https://github.com/MaryCassieOdie/MSDS688-Case-Study.git`). If the repo is private, copy the
   project folder to `My Drive/MSDS688-Case-Study` and set `REPO_DIR` in the first cell to that path.
2. Open each notebook in Colab (**File > Open notebook > GitHub**) and select a GPU runtime:
   **Runtime > Change runtime type > T4 GPU**.
3. Run the notebooks in order, 01 to 04. Each one starts by mounting Google Drive. Data and
   outputs go to `My Drive/MSDS688-Case-Study/`, so a runtime reset loses nothing, and finished
   steps such as downloads and trained models are reused on re-runs.

Approximate run times on a T4: 01 takes about 10 minutes (API calls and about 4,800 image
downloads). 02 takes 30 to 45 minutes for all three models. 03 and 04 take a few minutes each.

**Optional: official IUCN API.** Without a token, Red List categories come from the global IUCN
assessments that iNaturalist mirrors. To query IUCN directly, request a free token at
<https://api.iucnredlist.org/users/sign_up> and set `IUCN_API_TOKEN`. On Colab, store it under
the 🔑 Secrets tab and load it with `userdata.get`.

## Lite mode: quick local test on a CPU
Lite mode runs the whole pipeline (all 12 species, all three models, Grad-CAM and the rule
engine) on a laptop CPU in about 10 minutes (8 minutes in testing, including downloads). It uses the same code paths as a full run, but
with much less data:

| Setting | Full | Lite |
|---|---|---|
| Images per species | 400 | 30 |
| Image size | 224 px | 160 px |
| Epochs (head + fine-tune) | 8 + 12 | 3 + 2 |
| Occurrence sample | 14 years × 200 per species | 4 years × 50 per species |
| Needs-ID observations triaged | 60 | 12 |

Lite results are **not meaningful**. Use lite mode to check that everything runs, and use full
mode on a GPU for the numbers in the write-up. Lite data and outputs go to separate
`data_lite/` and `outputs_lite/` folders, so they never mix with a full run.

Install the dependencies once (Python 3.10 to 3.12):
```bash
pip install -r requirements.txt
```
Run all four notebooks headlessly. `--clean` starts from scratch, and the executed notebooks
are saved to `outputs_lite/executed/`:
```bash
python scripts/run_lite.py --clean
```
Or run them interactively:
```bash
cd notebooks && jupyter lab
```

In the notebooks, `LITE_MODE` in the first cell defaults to **lite when local and full on
Colab**. Set it to `True` or `False` to override, then restart the kernel. You can also set the
environment variable `POLLINATOR_LITE=1`.

TensorFlow no longer supports GPUs on native Windows, so a full local run on Windows uses the
CPU and is slow. Use Colab or WSL2 for full GPU training.

## Method summary
- **Data:** for each species, the most recent research-grade iNaturalist observations
  (`IMAGES_PER_CLASS = 400`), using only CC0, CC-BY and CC-BY-NC photos and wild organisms.
  Each observation contributes one photo. The split is 70/15/15, stratified by species.
- **Training:** each ImageNet backbone gets a new head (global average pooling, dropout, softmax).
  Phase 1 trains only the head. Phase 2 fine-tunes the top 40 backbone layers, with BatchNorm
  frozen. Training uses augmentation, class weights, early stopping and learning-rate reduction
  on plateau.
- **Evaluation:** accuracy, top-3 accuracy, macro-F1, recall on threatened species, recall and
  precision for the "threatened" flag, parameter count and inference latency.
- **Rule engine:** occurrence records are binned into a 1° grid. Six auditable rules (critical
  refuge, vulnerable stronghold, co-occurrence hotspot, possible local decline, under-surveyed
  area, pending expert review) add priority points and plain-language recommendations. A
  sensitivity analysis varies the grid size and the window that counts as "recent".

## Limitations
- The classifier is closed-set (12 species), so its outputs are suggestions for expert review,
  not automatic identifications.
- Citizen-science records are biased toward places people visit. The absence of records is not
  the absence of a species.
- Locations of threatened species are reported only at coarse resolution, and the project
  respects iNaturalist's coordinate obscuring.

## Data and licences
- Photos and observations come from [iNaturalist](https://www.inaturalist.org/). Each photo keeps
  its own licence, and credits are written to `data/ATTRIBUTION.csv`. Use is non-commercial and
  educational.
- Conservation categories come from the [IUCN Red List](https://www.iucnredlist.org/).
- The code is covered by the repository `LICENSE`.
