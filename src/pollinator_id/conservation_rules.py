"""Rule-based recommender for conservation investment locations.

A transparent, forward-chaining expert system. Occurrence records are binned
into a lat/lon grid; each grid cell is summarised (which species were seen,
how recently, and their IUCN category), then every rule in ``RULES`` is
evaluated against every cell. A fired rule adds priority points and a
plain-language recommendation, so each ranking can be traced back to the
exact rules that produced it.

Why a coarse grid? iNaturalist automatically obscures coordinates of many
threatened taxa to a ~0.2 deg box, so finer resolution would imply precision
the data does not have. The default 1 deg cell (~111 km N-S) is a regional
planning scale, not a site-selection tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from .iucn import SEVERITY, THREATENED


@dataclass
class RuleSettings:
    cell_deg: float = 1.0
    recent_years: int = 5               # "recent" = within this many years of reference_year
    reference_year: int = date.today().year
    min_recent_obs: int = 3             # evidence needed to call a population "present"
    min_historic_obs: int = 3           # evidence needed to call a population "formerly present"
    low_effort_obs: int = 20            # fewer total records than this = under-surveyed cell
    pending_confidence: float = 0.70    # model confidence for unverified threatened sightings


@dataclass
class CellContext:
    """Everything a rule may look at for one grid cell."""
    cell_id: str
    lat: float
    lon: float
    species: pd.DataFrame          # one row per species: n_recent, n_historic, iucn_code, ...
    total_obs: int
    all_obscured: bool
    pending: pd.DataFrame = field(default_factory=pd.DataFrame)  # model-flagged unverified sightings


@dataclass
class Rule:
    name: str
    points: int
    condition: Callable[[CellContext, RuleSettings], list[str]]  # returns triggering species
    recommendation: str


# ------------------------------------------------------------------ rule conditions
def _recent(ctx, s, codes):
    sp = ctx.species
    return sp[sp["iucn_code"].isin(codes) & (sp["n_recent"] >= s.min_recent_obs)]["label"].tolist()


def _critical_refuge(ctx, s):
    return _recent(ctx, s, {"CR", "EN"})


def _vulnerable_stronghold(ctx, s):
    return _recent(ctx, s, {"VU"})


def _cooccurrence(ctx, s):
    hits = _recent(ctx, s, THREATENED)
    return hits if len(hits) >= 2 else []


def _possible_decline(ctx, s):
    sp = ctx.species
    m = sp["iucn_code"].isin(THREATENED) & (sp["n_historic"] >= s.min_historic_obs) & (sp["n_recent"] == 0)
    return sp[m]["label"].tolist()


def _under_surveyed(ctx, s):
    sp = ctx.species
    threatened = sp[sp["iucn_code"].isin(THREATENED)]["label"].tolist()
    return threatened if threatened and ctx.total_obs < s.low_effort_obs else []


def _pending_review(ctx, s):
    p = ctx.pending
    if p.empty:
        return []
    m = p["pred_iucn_code"].isin(THREATENED) & (p["confidence"] >= s.pending_confidence)
    return sorted(p[m]["pred_label"].unique().tolist())


RULES: list[Rule] = [
    Rule("Critical refuge", 5, _critical_refuge,
         "Recent records of a Critically Endangered/Endangered species: prioritise habitat "
         "protection and pesticide reduction here."),
    Rule("Vulnerable stronghold", 3, _vulnerable_stronghold,
         "Persistent population of a Vulnerable species: fund habitat restoration "
         "(native flowering plants, nesting sites) to keep it from declining further."),
    Rule("Threatened co-occurrence hotspot", 3, _cooccurrence,
         "Two or more threatened species recently present: one investment benefits several species."),
    Rule("Possible local decline", 2, _possible_decline,
         "Threatened species recorded historically but not recently: commission targeted surveys "
         "to confirm status; candidate site for restoration or reintroduction."),
    Rule("Under-surveyed area", 1, _under_surveyed,
         "Threatened species present but few records overall: invest in monitoring "
         "(e.g. bioblitzes, Bumble Bee Watch) before larger commitments."),
    Rule("Pending expert review", 1, _pending_review,
         "Model flagged unverified sightings of threatened species: route these observations to "
         "expert identifiers to confirm."),
]


# ------------------------------------------------------------------ engine
def _cell_index(lat, lon, deg):
    return np.floor(np.asarray(lat) / deg).astype(int), np.floor(np.asarray(lon) / deg).astype(int)


def add_cells(df: pd.DataFrame, deg: float) -> pd.DataFrame:
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    r, c = _cell_index(df["latitude"], df["longitude"], deg)
    df["cell_row"], df["cell_col"] = r, c
    df["cell_id"] = [f"{a}_{b}" for a, b in zip(r, c)]
    return df


def evaluate(
    occurrences: pd.DataFrame,
    iucn_table: pd.DataFrame,
    settings: RuleSettings | None = None,
    pending: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score every grid cell.

    occurrences: columns label, latitude, longitude, observed_on (or year), coordinates_obscured
    iucn_table:  columns scientific_name, iucn_code
    pending:     optional model predictions on unverified observations with columns
                 latitude, longitude, pred_label, pred_iucn_code, confidence
    """
    s = settings or RuleSettings()
    occ = occurrences.copy()
    if "year" not in occ:
        occ["year"] = pd.to_datetime(occ["observed_on"], errors="coerce").dt.year
    occ = add_cells(occ.dropna(subset=["year"]), s.cell_deg)
    codes = iucn_table.set_index("scientific_name")["iucn_code"]
    occ["iucn_code"] = occ["label"].map(codes).fillna("NE")
    cutoff = s.reference_year - s.recent_years
    occ["is_recent"] = occ["year"] > cutoff

    pend = add_cells(pending, s.cell_deg) if pending is not None and len(pending) else pd.DataFrame()

    results = []
    for cell_id, g in occ.groupby("cell_id"):
        sp = (g.groupby(["label", "iucn_code"])
                .agg(n_recent=("is_recent", "sum"), n_total=("is_recent", "size"),
                     last_year=("year", "max"))
                .reset_index())
        sp["n_historic"] = sp["n_total"] - sp["n_recent"]
        ctx = CellContext(
            cell_id=cell_id,
            lat=(g["cell_row"].iloc[0] + 0.5) * s.cell_deg,
            lon=(g["cell_col"].iloc[0] + 0.5) * s.cell_deg,
            species=sp,
            total_obs=len(g),
            all_obscured=bool(g.get("coordinates_obscured", pd.Series([False])).all()),
            pending=pend[pend["cell_id"] == cell_id] if len(pend) else pd.DataFrame(),
        )
        fired, score, recs = [], 0, []
        for rule in RULES:
            hits = rule.condition(ctx, s)
            if hits:
                fired.append(f"{rule.name} ({', '.join(hits)})")
                score += rule.points
                recs.append(rule.recommendation)
        if not fired:
            continue
        threatened_here = sp[sp["iucn_code"].isin(THREATENED)]
        results.append({
            "cell_id": cell_id,
            "center_lat": ctx.lat,
            "center_lon": ctx.lon,
            "priority_score": score,
            "rules_fired": "; ".join(fired),
            "recommendations": " | ".join(recs),
            "threatened_species": ", ".join(
                f"{r.label} [{r.iucn_code}]" for r in threatened_here.sort_values(
                    "iucn_code", key=lambda c: c.map(SEVERITY), ascending=False).itertuples()),
            "max_severity": int(sp["iucn_code"].map(SEVERITY).max()),
            "total_records": ctx.total_obs,
            "coordinates_obscured": ctx.all_obscured,
        })
    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["priority_tier"] = pd.cut(out["priority_score"], [0, 3, 7, np.inf], labels=["Low", "Medium", "High"])
    return out.sort_values(["priority_score", "max_severity", "total_records"],
                           ascending=False).reset_index(drop=True)


def recommend_for_species(
    species: str,
    occurrences: pd.DataFrame,
    iucn_table: pd.DataFrame,
    settings: RuleSettings | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank cells for protecting one species: cells where the species is (or was)
    present, ordered by overall cell priority. Answers "where should we invest
    to protect species X?"."""
    s = settings or RuleSettings()
    cells = evaluate(occurrences, iucn_table, s)
    if cells.empty:
        return cells
    occ = occurrences.copy()
    if "year" not in occ:
        occ["year"] = pd.to_datetime(occ["observed_on"], errors="coerce").dt.year
    occ = add_cells(occ[occ["label"] == species], s.cell_deg)
    cutoff = s.reference_year - s.recent_years
    per_cell = occ.groupby("cell_id").agg(
        species_recent=("year", lambda y: int((y > cutoff).sum())),
        species_total=("year", "size"),
        species_last_year=("year", "max"),
    )
    ranked = cells.merge(per_cell, left_on="cell_id", right_index=True)
    return ranked.sort_values(["priority_score", "species_recent"], ascending=False).head(top_n).reset_index(drop=True)


def plot_priority_map(cells: pd.DataFrame, occurrences: pd.DataFrame | None = None, ax=None, title=None):
    """Static map of scored cells (marker size/colour = priority)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6.5))
    if occurrences is not None:
        ax.scatter(occurrences["longitude"], occurrences["latitude"], s=2, c="lightgray",
                   label="occurrence records", zorder=1)
    colors = {"High": "#b2182b", "Medium": "#ef8a62", "Low": "#fddbc7"}
    for tier in ["Low", "Medium", "High"]:
        sub = cells[cells["priority_tier"] == tier]
        ax.scatter(sub["center_lon"], sub["center_lat"], s=20 + 12 * sub["priority_score"],
                   c=colors[tier], edgecolor="k", linewidth=0.4, label=f"{tier} priority", zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "Conservation priority by grid cell")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def folium_map(cells: pd.DataFrame, cell_deg: float = 1.0):
    """Interactive Leaflet map (requires ``folium``, preinstalled on Colab)."""
    import folium

    m = folium.Map(location=[cells["center_lat"].mean(), cells["center_lon"].mean()], zoom_start=4,
                   tiles="cartodbpositron")
    colors = {"High": "#b2182b", "Medium": "#ef8a62", "Low": "#fddbc7"}
    for r in cells.itertuples():
        h = cell_deg / 2
        folium.Rectangle(
            bounds=[[r.center_lat - h, r.center_lon - h], [r.center_lat + h, r.center_lon + h]],
            color=colors[str(r.priority_tier)], fill=True, fill_opacity=0.55, weight=1,
            popup=folium.Popup(
                f"<b>{r.priority_tier} priority (score {r.priority_score})</b><br>"
                f"<i>{r.threatened_species}</i><br><br>{r.rules_fired}", max_width=350),
        ).add_to(m)
    return m
