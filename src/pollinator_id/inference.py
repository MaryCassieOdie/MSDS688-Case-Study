"""End-to-end identification: image -> species prediction -> IUCN Red List status.

This is the workflow from the case-study architecture slide, applied to the
real problem it targets: iNaturalist observations still waiting for a
species-level ID ("Needs ID"). The system suggests likely species for expert
review and flags any that would be threatened, so reviewers can prioritise
them.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from . import config
from .inaturalist import INatClient, observation_record
from .iucn import RedListReference
from .models import preprocess_fn


def load_image_from_url(url: str, img_size=config.IMG_SIZE) -> np.ndarray:
    from PIL import Image

    r = requests.get(url, timeout=30, headers={"User-Agent": config.USER_AGENT})
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB").resize(img_size)
    return np.asarray(img, dtype="float32")


def identify(model, backbone_name: str, raw_images: np.ndarray, redlist: RedListReference,
             top_k: int = 3, class_names=config.CLASS_NAMES) -> list[list[dict]]:
    """Top-k species suggestions with IUCN status for each raw [0, 255] image."""
    x = preprocess_fn(backbone_name)(np.array(raw_images, dtype="float32", copy=True))
    probs = model.predict(x, verbose=0)
    out = []
    for p in probs:
        top = np.argsort(p)[::-1][:top_k]
        out.append([{"species": class_names[i], "confidence": float(p[i]), **redlist.status(class_names[i])}
                    for i in top])
    return out


def fetch_needs_id_observations(client: INatClient, parent_taxon: str = "Bombus",
                                n: int = 60, place_id: int | None = None) -> pd.DataFrame:
    """Recent "Needs ID" observations identified only to a coarse level (e.g. genus
    *Bombus*) - exactly the backlog this system is meant to help clear."""
    results = client.get("taxa", q=parent_taxon, per_page=10)["results"]
    taxon = next(t for t in results if t["name"].lower() == parent_taxon.lower())
    # Observations of descendants of parent_taxon that are NOT yet at species level.
    filters = dict(taxon_id=taxon["id"], quality_grade="needs_id", photos="true",
                   photo_license=config.PHOTO_LICENSES, captive="false",
                   rank="genus,subgenus,complex")
    if place_id:
        filters["place_id"] = place_id
    rows = [observation_record(o, label="unknown") for o in client.iter_observations(max_results=n, **filters)]
    return pd.DataFrame([r for r in rows if r])


def triage(model, backbone_name: str, obs: pd.DataFrame, redlist: RedListReference,
           review_threshold: float = 0.5) -> pd.DataFrame:
    """Predict species for unidentified observations and rank them for expert review.

    Observations whose top prediction is a threatened species go to the top of
    the queue; confidence below ``review_threshold`` is marked "uncertain" -
    the system recommends, an expert decides.
    """
    images, keep = [], []
    for i, url in enumerate(obs["photo_url"]):
        try:
            images.append(load_image_from_url(url))
            keep.append(i)
        except Exception as exc:  # noqa: BLE001 - skip unreadable photos, keep going
            print(f"skipping {url}: {exc}")
    obs = obs.iloc[keep].reset_index(drop=True)
    preds = identify(model, backbone_name, np.stack(images), redlist)
    obs["pred_label"] = [p[0]["species"] for p in preds]
    obs["confidence"] = [p[0]["confidence"] for p in preds]
    obs["pred_iucn_code"] = [p[0]["iucn_code"] for p in preds]
    obs["pred_threatened"] = [p[0]["threatened"] for p in preds]
    obs["alternatives"] = [", ".join(f"{q['species']} ({q['confidence']:.0%})" for q in p[1:]) for p in preds]
    obs["review_note"] = np.where(obs["confidence"] >= review_threshold, "suggested ID", "uncertain")
    obs["image"] = list(images)  # raw [0, 255] arrays, for display / Grad-CAM
    return obs.sort_values(["pred_threatened", "confidence"], ascending=False).reset_index(drop=True)
