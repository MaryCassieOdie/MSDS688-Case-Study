"""Thin, rate-limited client for the public iNaturalist API (v1).

iNaturalist asks API users to stay under ~60 requests/minute and to keep
media downloads modest. ``INatClient`` sleeps between API calls, and image
downloads go through a small thread pool directly from the open-data S3
bucket that hosts openly licensed photos.

Docs: https://api.inaturalist.org/v1/docs/
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from . import config

API = "https://api.inaturalist.org/v1"


class INatClient:
    def __init__(self, min_interval: float = 1.1, timeout: float = 30.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_call = 0.0

    # ------------------------------------------------------------------ core
    def get(self, path: str, **params) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            self._last_call = time.monotonic()
            resp = self.session.get(f"{API}/{path}", params=params, timeout=self.timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(5 * (attempt + 1))  # back off on throttling / server errors
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    # ----------------------------------------------------------------- taxa
    def find_taxon(self, scientific_name: str) -> dict:
        """Return the species-rank taxon whose name matches exactly."""
        results = self.get("taxa", q=scientific_name, rank="species", per_page=30)["results"]
        for t in results:
            if t["name"].lower() == scientific_name.lower():
                return t
        raise LookupError(f"No exact iNaturalist species match for {scientific_name!r}")

    def taxon_details(self, taxon_id: int) -> dict:
        """Full taxon record, including the ``conservation_statuses`` list."""
        return self.get(f"taxa/{taxon_id}")["results"][0]

    # ---------------------------------------------------------- observations
    def iter_observations(self, max_results: int | None = None, newest_first: bool = True,
                          **filters) -> Iterator[dict]:
        """Page through observations with ``id_below``/``id_above`` (iNat's
        recommended method for deep paging; ``page`` is capped at 10k results)."""
        order = "desc" if newest_first else "asc"
        cursor_key = "id_below" if newest_first else "id_above"
        params = dict(per_page=200, order_by="id", order=order, **filters)
        cursor, n = None, 0
        while True:
            if cursor is not None:
                params[cursor_key] = cursor
            batch = self.get("observations", **params)["results"]
            if not batch:
                return
            for obs in batch:
                yield obs
                n += 1
                if max_results is not None and n >= max_results:
                    return
            cursor = batch[-1]["id"]


# ------------------------------------------------------------------ helpers
def observation_record(obs: dict, label: str) -> dict | None:
    """Flatten an observation into one metadata row (first photo only, so each
    image is an independent observation and the train/test split cannot leak
    photos of the same individual)."""
    photos = obs.get("photos") or []
    if not photos:
        return None
    photo = photos[0]
    lat = lon = None
    if obs.get("location"):
        lat, lon = (float(v) for v in obs["location"].split(","))
    return {
        "observation_id": obs["id"],
        "label": label,
        "taxon_name": (obs.get("taxon") or {}).get("name"),
        "photo_id": photo["id"],
        "photo_url": photo["url"].replace("square", config.PHOTO_SIZE),
        "license": photo.get("license_code"),
        "attribution": photo.get("attribution"),
        "latitude": lat,
        "longitude": lon,
        # Threatened taxa are often auto-obscured (~0.2 deg box); keep the flag so
        # downstream spatial analysis can use a coarse grid.
        "coordinates_obscured": bool(obs.get("obscured")),
        "observed_on": obs.get("observed_on"),
        "quality_grade": obs.get("quality_grade"),
        "place_guess": obs.get("place_guess"),
        "url": f"https://www.inaturalist.org/observations/{obs['id']}",
    }


def collect_species_metadata(
    client: INatClient,
    scientific_name: str,
    n_images: int = config.IMAGES_PER_CLASS,
    quality_grade: str = "research",
) -> pd.DataFrame:
    taxon = client.find_taxon(scientific_name)
    rows = []
    for obs in client.iter_observations(
        max_results=n_images,
        taxon_id=taxon["id"],
        quality_grade=quality_grade,
        photos="true",
        photo_license=config.PHOTO_LICENSES,
        # Exclude captive/cultivated records: we want wild animals in the field.
        captive="false",
    ):
        rec = observation_record(obs, scientific_name)
        # taxon_id filter includes subspecies; keep the species label regardless.
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)


def _download_one(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def download_images(df: pd.DataFrame, image_dir: Path = config.IMAGE_DIR, workers: int = 4) -> pd.DataFrame:
    """Download each row's photo to ``image_dir/<label>/<photo_id>.jpg``.

    Adds a ``filepath`` column (relative to ``image_dir``) and drops rows whose
    download failed. Already-downloaded files are skipped, so it is safe to
    re-run after an interruption.
    """
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    df = df.copy()
    df["filepath"] = [
        f"{label.replace(' ', '_')}/{pid}.jpg" for label, pid in zip(df["label"], df["photo_id"])
    ]
    for label in df["label"].unique():
        (image_dir / label.replace(" ", "_")).mkdir(parents=True, exist_ok=True)

    ok = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, url, image_dir / fp, session): fp
            for url, fp in zip(df["photo_url"], df["filepath"])
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ok[futures[fut]] = fut.result()
            if i % 250 == 0:
                print(f"  downloaded {i}/{len(futures)}")
    df = df[df["filepath"].map(ok)].reset_index(drop=True)
    return df


def collect_occurrences(
    client: INatClient,
    scientific_name: str,
    years: range,
    per_year: int = 200,
) -> pd.DataFrame:
    """Coordinates-only occurrence sample, stratified by year.

    Used by the conservation rule engine. Sampling up to ``per_year`` research-
    grade records from *every* year keeps abundant species from being
    represented only by their most recent records, so historic-vs-recent
    comparisons (possible local declines) are meaningful.
    """
    taxon = client.find_taxon(scientific_name)
    rows = []
    for year in years:
        for obs in client.iter_observations(
            max_results=per_year, taxon_id=taxon["id"], quality_grade="research",
            year=year, captive="false",
        ):
            if not obs.get("location"):
                continue
            lat, lon = (float(v) for v in obs["location"].split(","))
            rows.append({
                "observation_id": obs["id"],
                "label": scientific_name,
                "latitude": lat,
                "longitude": lon,
                "coordinates_obscured": bool(obs.get("obscured")),
                "observed_on": obs.get("observed_on"),
                "year": year,
            })
    return pd.DataFrame(rows)
