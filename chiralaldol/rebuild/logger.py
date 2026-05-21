"""Centralized logging and row-level audit infrastructure for V3 rebuild."""

import logging
import sys
from pathlib import Path

import pandas as pd

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console + optional file output."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.flush = lambda: fh.stream.flush()  # force flush on each write
        handlers.append(fh)
    logging.basicConfig(level=level, format=LOG_FMT, handlers=handlers, force=True)


class AuditTracker:
    """Row-level audit tracker accumulating per-row metadata across pipeline steps."""

    def __init__(self, n_rows: int):
        self.df = pd.DataFrame({"original_index": range(n_rows)})
        self._deletion_reasons: dict[int, str] = {}

    def add_column(self, name: str, values) -> None:
        self.df[name] = values

    def mark_deleted_by_oi(self, original_indices, reason: str) -> None:
        """Record deletion reason by original_index values (safe across filtered DataFrames)."""
        for oi in original_indices:
            if oi not in self._deletion_reasons:
                self._deletion_reasons[oi] = reason

    def get_deletion_reasons(self) -> pd.Series:
        """Return Series indexed by original_index with deletion reason strings."""
        return pd.Series(self._deletion_reasons, name="deletion_reason")

    def finalize(self, kept_original_indices, evans_original_indices) -> pd.DataFrame:
        """Build final audit DataFrame covering all original rows."""
        kept_set = set(kept_original_indices)
        evans_set = set(evans_original_indices)
        reasons = self.get_deletion_reasons()

        final_set = []
        deletion_reason = []
        for oi in self.df["original_index"]:
            if oi in kept_set:
                if oi in evans_set:
                    final_set.append("evans_v3")
                else:
                    final_set.append("non_evans_v3")
                deletion_reason.append("")
            else:
                final_set.append("deleted")
                deletion_reason.append(reasons.get(oi, "unknown"))

        self.df["final_set"] = final_set
        self.df["deletion_reason"] = deletion_reason
        return self.df
