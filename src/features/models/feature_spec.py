from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True, slots=True)
class FeatureParameter:
    name: str
    default: Any

    def resolve(self, config_source: object) -> Any:
        return getattr(config_source, self.name, self.default)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    block: str
    formula_template: str
    description: str = ""
    params: tuple[FeatureParameter, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    shift: int = 0

    def resolved_params(self, config_source: object) -> dict[str, Any]:
        return {
            parameter.name: parameter.resolve(config_source)
            for parameter in self.params
        }

    def resolved_formula(self, config_source: object) -> str:
        return self.formula_template.format_map(_SafeFormatDict(self.resolved_params(config_source)))

    def to_payload(self, config_source: object) -> dict[str, Any]:
        return {
            "name": self.name,
            "block": self.block,
            "description": self.description,
            "formula_template": self.formula_template,
            "resolved_formula": self.resolved_formula(config_source),
            "params": self.resolved_params(config_source),
            "inputs": list(self.inputs),
            "dependencies": list(self.dependencies),
            "shift": self.shift,
        }


def feature_param(name: str, default: Any) -> FeatureParameter:
    return FeatureParameter(name=name, default=default)


def feature_spec(
    name: str,
    block: str,
    formula_template: str,
    *,
    description: str = "",
    params: tuple[FeatureParameter, ...] = (),
    inputs: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    shift: int = 0,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        block=block,
        formula_template=formula_template,
        description=description,
        params=params,
        inputs=inputs,
        dependencies=dependencies,
        shift=shift,
    )


def serialize_feature_specs(
    specs: Mapping[str, FeatureSpec],
    config_source: object,
) -> list[dict[str, Any]]:
    return [
        specs[feature_name].to_payload(config_source)
        for feature_name in sorted(specs)
    ]
