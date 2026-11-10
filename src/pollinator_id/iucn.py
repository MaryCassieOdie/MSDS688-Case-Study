"""Cross-reference species with the IUCN Red List of Threatened Species.

Two sources are supported:

1. **IUCN Red List API v4** (authoritative) - used when the environment
   variable ``IUCN_API_TOKEN`` is set. Request a free token at
   https://api.iucnredlist.org/users/sign_up
2. **iNaturalist taxon records** (default, no token needed). iNaturalist
   mirrors global IUCN assessments in each taxon's ``conservation_statuses``
   list under the authority ``"IUCN Red List"``.

A species with no global IUCN assessment is reported as ``NE`` (Not Evaluated).
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from . import config
from .inaturalist import INatClient

IUCN_CATEGORIES = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
}
THREATENED = {"CR", "EN", "VU"}

# Ordinal severity used for sorting and by the conservation rule engine.
SEVERITY = {"NE": 0, "DD": 0, "LC": 1, "NT": 2, "VU": 3, "EN": 4, "CR": 5, "EW": 6, "EX": 7}

# iNaturalist encodes IUCN-equivalent levels numerically.
_INAT_IUCN_CODES = {0: "NE", 5: "DD", 10: "LC", 20: "NT", 30: "VU", 40: "EN", 50: "CR", 60: "EW", 70: "EX"}


def _from_inaturalist(client: INatClient, scientific_name: str) -> dict:
    taxon = client.find_taxon(scientific_name)
    details = client.taxon_details(taxon["id"])
    code, url = "NE", None
    for s in details.get("conservation_statuses") or []:
        # Only the *global* IUCN assessment; regional ones have a place attached.
        if s.get("authority") == "IUCN Red List" and s.get("place") is None:
            raw = (s.get("status") or "").upper()
            code = raw if raw in IUCN_CATEGORIES else _INAT_IUCN_CODES.get(s.get("iucn"), "NE")
            url = s.get("url")
            break
    return {
        "taxon_id": taxon["id"],
        "common_name": taxon.get("preferred_common_name"),
        "iucn_code": code,
        "assessment_url": url,
        "source": "iNaturalist (IUCN Red List mirror)",
    }


def _from_iucn_api(scientific_name: str, token: str) -> dict | None:
    genus, species = scientific_name.split()[:2]
    resp = requests.get(
        "https://api.iucnredlist.org/api/v4/taxa/scientific_name",
        params={"genus_name": genus, "species_name": species},
        headers={
            "Authorization": token if token.lower().startswith("bearer ") else f"Bearer {token}",
            "User-Agent": config.USER_AGENT,
        },
        timeout=30,
    )
    if resp.status_code == 404:
        return {"iucn_code": "NE", "assessment_url": None, "source": "IUCN Red List API v4"}
    resp.raise_for_status()
    assessments = resp.json().get("assessments") or []
    latest = [a for a in assessments if a.get("latest")] or assessments
    # Prefer the global (not regional) assessment.
    glob = [a for a in latest if "global" in str(a.get("scopes", "")).lower()] or latest
    if not glob:
        return {"iucn_code": "NE", "assessment_url": None, "source": "IUCN Red List API v4"}
    a = glob[0]
    return {
        "iucn_code": (a.get("red_list_category_code") or "NE").upper(),
        "assessment_url": a.get("url"),
        "assessment_year": a.get("year_published"),
        "source": "IUCN Red List API v4",
    }


def lookup_statuses(species_names: list[str], client: INatClient | None = None) -> pd.DataFrame:
    """Return one row per species with its IUCN category and metadata."""
    client = client or INatClient()
    token = os.environ.get("IUCN_API_TOKEN")
    rows = []
    for name in species_names:
        rec = {"scientific_name": name, **_from_inaturalist(client, name)}
        if token:
            try:
                official = _from_iucn_api(name, token)
                if official:
                    rec.update(official)
            except requests.RequestException as exc:  # keep iNat fallback
                print(f"IUCN API lookup failed for {name}: {exc}; using iNaturalist mirror")
        rec["iucn_category"] = IUCN_CATEGORIES.get(rec["iucn_code"], rec["iucn_code"])
        rec["threatened"] = rec["iucn_code"] in THREATENED
        rows.append(rec)
    return pd.DataFrame(rows)


class RedListReference:
    """In-memory lookup table used at prediction time."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.set_index("scientific_name")

    @classmethod
    def from_csv(cls, path=config.IUCN_CSV) -> "RedListReference":
        return cls(pd.read_csv(path))

    def status(self, scientific_name: str) -> dict:
        if scientific_name not in self.table.index:
            return {"iucn_code": "NE", "iucn_category": "Not Evaluated", "threatened": False}
        row = self.table.loc[scientific_name]
        return {
            "iucn_code": row["iucn_code"],
            "iucn_category": row["iucn_category"],
            "threatened": bool(row["threatened"]),
        }
