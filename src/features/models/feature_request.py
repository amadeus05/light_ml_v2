from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ALL_PROFILE_SENTINEL = "__all__"


@dataclass(frozen=True, slots=True)
class ResolvedFeatureRequest:
    profile: str
    active_features: tuple[str, ...]
    active_blocks: tuple[str, ...]


def resolve_feature_request(
    raw_request: dict | None,
    profile_map: dict[str, object] | None,
    block_features: dict[str, set[str]],
) -> ResolvedFeatureRequest:
    available_features = set().union(*block_features.values()) if block_features else set()
    profiles = dict(profile_map or {})
    profiles.setdefault("all", ALL_PROFILE_SENTINEL)
    profiles.setdefault("empty", [])

    request = dict(raw_request or {})
    profile_name = str(request.get("profile", "all"))
    include_features = _to_string_set(request.get("include_features"))
    exclude_features = _to_string_set(request.get("exclude_features"))
    exclude_blocks = _to_string_set(request.get("exclude_blocks"))

    if profile_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown feature profile '{profile_name}'. Known profiles: {known}")

    profile_value = profiles[profile_name]
    if profile_value == ALL_PROFILE_SENTINEL:
        active_features = set(available_features)
    else:
        active_features = _to_string_set(profile_value)

    unknown_profile_features = sorted(active_features - available_features)
    if unknown_profile_features:
        raise ValueError(
            "Feature profile contains unknown features: " + ", ".join(unknown_profile_features)
        )

    unknown_includes = sorted(include_features - available_features)
    if unknown_includes:
        raise ValueError("FEATURE_BUILD_REQUEST.include_features contains unknown features: " + ", ".join(unknown_includes))

    unknown_excludes = sorted(exclude_features - available_features)
    if unknown_excludes:
        raise ValueError("FEATURE_BUILD_REQUEST.exclude_features contains unknown features: " + ", ".join(unknown_excludes))

    unknown_blocks = sorted(exclude_blocks - set(block_features))
    if unknown_blocks:
        raise ValueError("FEATURE_BUILD_REQUEST.exclude_blocks contains unknown blocks: " + ", ".join(unknown_blocks))

    active_features.update(include_features)
    active_features.difference_update(exclude_features)
    for block_name in exclude_blocks:
        active_features.difference_update(block_features[block_name])

    active_blocks = tuple(
        block_name
        for block_name in sorted(block_features)
        if block_features[block_name].intersection(active_features)
    )
    return ResolvedFeatureRequest(
        profile=profile_name,
        active_features=tuple(sorted(active_features)),
        active_blocks=active_blocks,
    )


def _to_string_set(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    if isinstance(values, Iterable):
        return {str(value) for value in values}
    raise ValueError(f"Expected iterable of feature names, got: {type(values)!r}")
