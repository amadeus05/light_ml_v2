from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMN = "Target"
TARGET_LONG_COLUMN = "TargetLong"
TARGET_SHORT_COLUMN = "TargetShort"

BARRIER_STOP_PCT_COLUMN = "barrier_stop_pct"
BARRIER_TAKE_PCT_COLUMN = "barrier_take_pct"
LABEL_HORIZON_BARS_COLUMN = "label_horizon_bars"

LONG_LABEL_PNL_COLUMN = "long_label_pnl"
SHORT_LABEL_PNL_COLUMN = "short_label_pnl"
LONG_EXIT_REASON_COLUMN = "long_exit_reason"
SHORT_EXIT_REASON_COLUMN = "short_exit_reason"

DUAL_TARGET_VALUES = (-1, 0, 1)
BINARY_TARGET_VALUES = (0, 1)
EXIT_REASONS = ("TP", "SL", "TIMEOUT")

LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {0: -1, 1: 1}


@dataclass(frozen=True, slots=True)
class LabelingSchema:
    target: str = TARGET_COLUMN
    target_long: str = TARGET_LONG_COLUMN
    target_short: str = TARGET_SHORT_COLUMN
    barrier_stop_pct: str = BARRIER_STOP_PCT_COLUMN
    barrier_take_pct: str = BARRIER_TAKE_PCT_COLUMN
    label_horizon_bars: str = LABEL_HORIZON_BARS_COLUMN
    long_label_pnl: str = LONG_LABEL_PNL_COLUMN
    short_label_pnl: str = SHORT_LABEL_PNL_COLUMN
    long_exit_reason: str = LONG_EXIT_REASON_COLUMN
    short_exit_reason: str = SHORT_EXIT_REASON_COLUMN

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.target, self.target_long, self.target_short)

    @property
    def barriers(self) -> tuple[str, ...]:
        return (self.barrier_stop_pct, self.barrier_take_pct, self.label_horizon_bars)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return (
            self.long_label_pnl,
            self.short_label_pnl,
            self.long_exit_reason,
            self.short_exit_reason,
        )

    @property
    def output_columns(self) -> list[str]:
        return [*self.barriers, *self.diagnostics]

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset((*self.targets, *self.barriers, *self.diagnostics))


LABELING_SCHEMA = LabelingSchema()

MODEL_TARGET_COLUMNS = set(LABELING_SCHEMA.targets)
TARGET_OUTPUT_COLUMNS = list(LABELING_SCHEMA.targets)
LABELING_CONTRACT_COLUMNS = set(LABELING_SCHEMA.barriers)
LABEL_DIAGNOSTIC_COLUMNS = set(LABELING_SCHEMA.diagnostics)
LABELING_OUTPUT_COLUMNS = LABELING_SCHEMA.output_columns

__all__ = [
    "BARRIER_STOP_PCT_COLUMN",
    "BARRIER_TAKE_PCT_COLUMN",
    "BINARY_TARGET_VALUES",
    "CLASS_TO_LABEL",
    "DUAL_TARGET_VALUES",
    "EXIT_REASONS",
    "LABEL_DIAGNOSTIC_COLUMNS",
    "LABEL_HORIZON_BARS_COLUMN",
    "LABEL_TO_CLASS",
    "LABELING_CONTRACT_COLUMNS",
    "LABELING_OUTPUT_COLUMNS",
    "LABELING_SCHEMA",
    "LabelingSchema",
    "LONG_EXIT_REASON_COLUMN",
    "LONG_LABEL_PNL_COLUMN",
    "MODEL_TARGET_COLUMNS",
    "SHORT_EXIT_REASON_COLUMN",
    "SHORT_LABEL_PNL_COLUMN",
    "TARGET_COLUMN",
    "TARGET_LONG_COLUMN",
    "TARGET_OUTPUT_COLUMNS",
    "TARGET_SHORT_COLUMN",
]
