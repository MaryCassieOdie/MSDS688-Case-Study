"""Endangered pollinator identification system (MSDS688 case study).

Pipeline: iNaturalist images -> TensorFlow CNN -> predicted species ->
IUCN Red List status -> rule-based conservation recommendations.
"""
from . import config  # noqa: F401

__version__ = "1.0.0"
