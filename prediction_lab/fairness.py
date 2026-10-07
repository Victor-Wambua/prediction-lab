"""Provably-fair verification research (separate from prediction on purpose).

Only uses values the game itself shows a player in its "Provably fair" dialog
for a FINISHED round (server seed revealed after the round, client seeds, the
combined hash, the result). Nothing here tries to obtain hidden secrets.

The exact derivation the game provider uses is NOT hard-coded as fact. Several candidate
formulas are implemented as hypotheses; `verify_round` reports which of them,
if any, reproduce the displayed multiplier. Record several real rounds and see
which hypothesis survives all of them.

Note: a verified derivation proves the result came from those seeds. It does not
make future rounds predictable, because the next server seed is secret until the
round ends.
"""
import hashlib
import math


def calculate_hash(server_seed: str, client_seeds: list[str], algo="sha512", sep="") -> str:
    combined = sep.join([server_seed, *client_seeds])
    return hashlib.new(algo, combined.encode()).hexdigest()


def _from_hex13(hex_hash: str, rtp: float) -> float:
    """x = rtp / U with U from the first 52 bits; floored to 2dp; min 1.00."""
    h = int(hex_hash[:13], 16)
    e = 2 ** 52
    x = rtp * e / (e - h)
    return max(1.0, math.floor(x * 100) / 100)


def _bustabit(hex_hash: str) -> float:
    """Classic bustabit-style formula (instant 1.00x when hash % 33 == 0)."""
    if int(hex_hash, 16) % 33 == 0:
        return 1.0
    h = int(hex_hash[:13], 16)
    e = 2 ** 52
    return math.floor((100 * e - h) / (e - h)) / 100


CANDIDATE_FORMULAS = {
    "hex13_rtp97": lambda hx: _from_hex13(hx, 0.97),
    "hex13_rtp99": lambda hx: _from_hex13(hx, 0.99),
    "bustabit": _bustabit,
}


def calculate_multiplier(hex_hash: str, formula: str) -> float:
    return CANDIDATE_FORMULAS[formula](hex_hash)


def compare_result(computed: float, displayed: float, tol=0.005) -> bool:
    return abs(computed - displayed) <= tol


def verify_round(server_seed: str, client_seeds: list[str], displayed_multiplier: float,
                 displayed_hash: str | None = None, server_seed_hash: str | None = None) -> dict:
    """Try every (combination, formula) hypothesis; return what matched."""
    out = {"server_seed_hash_ok": None, "hash_matches": [], "formula_matches": []}
    if server_seed_hash:
        out["server_seed_hash_ok"] = {
            a: hashlib.new(a, server_seed.encode()).hexdigest() == server_seed_hash.lower()
            for a in ("sha256", "sha512")}
    for algo in ("sha512", "sha256"):
        for sep in ("", ":", "-"):
            hx = calculate_hash(server_seed, client_seeds, algo, sep)
            combo = f"{algo}(sep={sep!r})"
            if displayed_hash and hx == displayed_hash.lower():
                out["hash_matches"].append(combo)
            for name in CANDIDATE_FORMULAS:
                if compare_result(calculate_multiplier(hx, name), displayed_multiplier):
                    out["formula_matches"].append(f"{combo} + {name}")
    if displayed_hash:  # also test formulas directly on the hash the game displays
        for name in CANDIDATE_FORMULAS:
            if compare_result(calculate_multiplier(displayed_hash.lower(), name), displayed_multiplier):
                out["formula_matches"].append(f"displayed_hash + {name}")
    return out
