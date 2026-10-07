"""Pure logic for turning history-strip snapshots into new rounds (no browser).

The site shows the last N results, newest first. It may not expose round IDs, and
values repeat (1.00x happens constantly), so values alone cannot identify a round.

Instead we align SEQUENCES: if the stored tail (newest first) is T and the new
snapshot is S, then S = [k new rounds] + T[:len(S)-k]. We find the smallest k for
which S[k:] matches the start of T over at least `min_overlap` rounds. If no k
works, continuity cannot be proven and the caller opens a new segment instead of
guessing (a "gap"). This survives refreshes and restarts because T comes from the
database, not from memory.
"""
import re
from dataclasses import dataclass

_MULT_RE = re.compile(r"^\s*(\d+(?:\.\d{1,2})?)\s*[xX×]\s*$")


def parse_multiplier(text: str, max_value: float = 1_000_000.0) -> float | None:
    """Strictly parse one history item such as '1.37x' or '1,234.50x'. None if invalid."""
    if text is None:
        return None
    m = _MULT_RE.match(text.replace(",", "").replace(" ", " "))
    if not m:
        return None
    v = float(m.group(1))
    if not (1.0 <= v <= max_value):
        return None
    return v


def to_cents(values) -> list[int]:
    return [int(round(v * 100)) for v in values]


@dataclass
class Alignment:
    new_count: int | None   # None => gap (no provable continuity)
    ambiguous: bool = False  # several k fit (e.g. long constant runs); smallest chosen
    overlap: int = 0


def align(snapshot_newest_first, tail_newest_first, min_overlap: int = 8) -> Alignment:
    S = to_cents(snapshot_newest_first)
    T = to_cents(tail_newest_first)
    if not T:
        return Alignment(None)
    need = max(1, min(min_overlap, len(T)))
    candidates = []
    for k in range(len(S) + 1):
        m = min(len(S) - k, len(T))
        if m < need:
            break
        if S[k:k + m] == T[:m]:
            candidates.append((k, m))
    if not candidates:
        return Alignment(None)
    k, m = candidates[0]
    return Alignment(k, ambiguous=len(candidates) > 1, overlap=m)


def parse_snapshot(texts, max_value=1_000_000.0) -> list[float] | None:
    """All items must parse; a single bad item rejects the snapshot (fail loudly)."""
    values = [parse_multiplier(t, max_value) for t in texts]
    if not values or any(v is None for v in values):
        return None
    return values
