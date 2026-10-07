from aviator_lab import fairness


def test_verify_round_identifies_the_generating_hypothesis():
    server, clients = "server-seed-abc", ["c1", "c2", "c3"]
    hx = fairness.calculate_hash(server, clients, "sha512", "")
    x = fairness.calculate_multiplier(hx, "hex13_rtp97")
    out = fairness.verify_round(server, clients, x, displayed_hash=hx)
    assert "sha512(sep='')" in out["hash_matches"]
    assert "sha512(sep='') + hex13_rtp97" in out["formula_matches"]


def test_formulas_are_valid_multipliers():
    for i in range(200):
        hx = fairness.calculate_hash(f"s{i}", ["a"])
        for name in fairness.CANDIDATE_FORMULAS:
            v = fairness.calculate_multiplier(hx, name)
            assert v >= 1.0 and round(v, 2) == v
