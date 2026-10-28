"""Scenario suites and the runner that judges episodes on what they committed."""

from .compositional import V6_FAMILIES  # registers the v6 families in FAMILIES
from .runner import Verdict, run_suite, save, summary
from .scenarios import (
    DEFAULT_FAMILIES,
    FAMILIES,
    HARD_FAMILIES,
    STRESS_FAMILIES,
    V1_FAMILIES,
    V2_FAMILIES,
    V3_FAMILIES,
    V5_FAMILIES,
    Scenario,
    build_suite,
)

__all__ = ["DEFAULT_FAMILIES", "FAMILIES", "HARD_FAMILIES", "STRESS_FAMILIES", "V1_FAMILIES", "V2_FAMILIES", "V3_FAMILIES", "V5_FAMILIES", "V6_FAMILIES", "Scenario", "Verdict", "build_suite", "run_suite", "save", "summary"]
