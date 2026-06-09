from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.models.feature_spec import FeatureSpec


@dataclass(frozen=True, slots=True)
class FeaturePipelineResult:
    feature_map: dict[str, pd.DataFrame]
    feature_columns: tuple[str, ...]
    active_blocks: tuple[str, ...]
    profile_name: str
    feature_specs: dict[str, FeatureSpec]
