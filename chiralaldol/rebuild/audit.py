"""Row-level audit tracker for the V4 rebuild pipeline."""

import pandas as pd
from collections import defaultdict


class AuditTracker:
    """Track why each row was kept or dropped across pipeline steps."""

    def __init__(self, n_total: int):
        self.n_total = n_total
        # step_name -> list of (original_idx, reason)
        self._drops: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self._step_counts: list[tuple[str, int]] = []

    def record_drop(self, step: str, indices, reason: str):
        """Record dropped row indices with reason."""
        for idx in indices:
            self._drops[step].append((idx, reason))

    def record_step(self, step: str, n_remaining: int):
        """Record row count after a step."""
        self._step_counts.append((step, n_remaining))

    def summary_df(self) -> pd.DataFrame:
        """Per-step summary table."""
        rows = []
        prev = self.n_total
        for step, count in self._step_counts:
            dropped = prev - count
            rows.append({
                "step": step,
                "rows_in": prev,
                "rows_out": count,
                "dropped": dropped,
                "pct_dropped": f"{dropped / max(prev, 1) * 100:.1f}%",
            })
            prev = count
        return pd.DataFrame(rows)

    def row_audit_df(self) -> pd.DataFrame:
        """Row-level audit: each dropped row with step and reason."""
        rows = []
        for step, entries in self._drops.items():
            for idx, reason in entries:
                rows.append({"original_idx": idx, "step": step, "reason": reason})
        return pd.DataFrame(rows)

    def print_summary(self):
        """Print step summary to stdout."""
        df = self.summary_df()
        print("\n" + "=" * 60)
        print("PIPELINE AUDIT SUMMARY")
        print("=" * 60)
        for _, row in df.iterrows():
            print(f"  {row['step']:40s}  {row['rows_in']:>6d} -> {row['rows_out']:>6d}  "
                  f"(-{row['dropped']:>5d}, {row['pct_dropped']:>6s})")
        print("=" * 60)
