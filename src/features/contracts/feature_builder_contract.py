from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import pandas as pd

from src.features.models.feature_context import FeatureContext
from src.features.models.feature_spec import FeatureSpec


class FeatureBuilderContract(ABC):
    block_name: str
    FEATURE_SPECS: Mapping[str, FeatureSpec] = {}

    def provides(self) -> set[str]:
        return set(self.feature_specs())

    def feature_specs(self) -> dict[str, FeatureSpec]:
        specs = dict(self.FEATURE_SPECS)
        if not specs:
            raise NotImplementedError(f"{self.__class__.__name__} must define FEATURE_SPECS.")

        invalid_names = [
            feature_name
            for feature_name, spec in specs.items()
            if spec.name != feature_name
        ]
        if invalid_names:
            raise ValueError(
                f"{self.__class__.__name__} contains mismatched FeatureSpec names: {', '.join(sorted(invalid_names))}"
            )

        invalid_blocks = [
            feature_name
            for feature_name, spec in specs.items()
            if spec.block != self.block_name
        ]
        if invalid_blocks:
            raise ValueError(
                f"{self.__class__.__name__} contains FeatureSpecs with wrong block: {', '.join(sorted(invalid_blocks))}"
            )
        return specs

    def describe_features(self, requested_features: set[str] | None = None) -> dict[str, FeatureSpec]:
        specs = self.feature_specs()
        if requested_features is None:
            return specs
        return {
            feature_name: spec
            for feature_name, spec in specs.items()
            if feature_name in requested_features
        }

    @abstractmethod
    def build(self, context: FeatureContext, requested_features: set[str]) -> pd.DataFrame:
        raise NotImplementedError
