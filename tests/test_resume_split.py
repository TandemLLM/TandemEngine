"""tools/resume_split.py (R83's G2) on a tiny random Hy4-flash model, on the CPU, where every row count takes one path
(the torch reference: there is no class switch to see there):

  1  the served pass (v2): a prompt read cold equals the same prompt read as two pieces at every k, through
     `prefill` and through `forward`, the one-row tail included: Hy4-flash sets `Capabilities.prompt_rows` (R83, T5b),
     so the server reads a one-row piece on the prompt pass, not as a decode step (whose row pass differs from the
     prompt pass in five places, the rowinv review's M1). The pass before (v1-pair) is reported only: on the CPU it is
     unequal at k 16. The record is family_check's shape
                                                                   test_cold_against_two_pieces_on_the_cpu
  2  the tool sees a difference: a state row changed between the two pieces is reported, with its layer, and the
     greedy tokens are compared                                    test_a_changed_row_is_reported
  3  the k list holds both sides of each class line and the run A prompt's cuts [349, 74], [390, 33]
                                                                   test_the_k_list
  4  the decode-tail arm (G2d, the end read again): ids[:k] on the prompt pass and ids[k:] by decode steps or chain
     blocks hold the exact parts (length, token history, the group-key mirror, the pooled rows) with layer 0 within
     2^-7; read again on the prompt pass from k, every section and view, the logits and the greedy tokens equal the
     cold read's                                                   test_the_decode_tail_arm

CPU. Run: python tests/test_resume_split.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine.models.hy4_flash.engine import Hy4FlashEngine  # noqa: E402
from engine.models.hy4_flash.serving import Hy4FlashServing  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402
from tools import resume_split as RS  # noqa: E402

_E: list = []


def _setup(max_len: int = 160):
    if not _E:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        eng = Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=max_len)
        _E.append((eng, Hy4FlashServing(eng, max_len)))
    return _E[0]


def _ids(n: int, seed: int = 3) -> list:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 60, (n,), generator=g).tolist()


#: the cases known unequal on the CPU at this tree (none of the served pass: before T5b the one-row tail through
#: `forward` was a decode step)
KNOWN_UNEQUAL: set = set()


def test_cold_against_two_pieces_on_the_cpu():
    eng, srv = _setup()
    cases = RS.run(eng, srv, {"p": _ids(40)}, ["v2", "v1-pair"], list(RS.VIAS), lambda n: [1, 7, 16, 33, n - 1], 4,
                   lambda line: None)
    assert len(cases) == 2 * 2 * 5, len(cases)
    v2 = {k: v for k, v in cases.items() if ".v2." in k}
    bad = sorted(k for k, v in v2.items() if not v["equal"])
    assert set(bad) == KNOWN_UNEQUAL, (bad, {k: (v2[k]["state_differing"], v2[k]["logits_max_abs"]) for k in bad})
    tail = cases["p.v2.forward.k39"]
    assert tail["piece_classes"][1] == "prompt order (S1 w8)", tail["piece_classes"]
    assert RS.piece_class(1, 39, "forward") == "decode step (row pass)"         # a family without the capability
    rec = RS.record(cases, "hy4_flash", "tiny")
    assert rec["pin"] == "g2" and set(rec["cases"]) == set(cases)
    v1 = sorted(k for k, v in cases.items() if ".v1-pair." in k and not v["equal"])
    return f"v2: {len(v2) - len(bad)} of {len(v2)} equal, the one-row tail through forward included; v1-pair unequal: {v1}"


def test_a_changed_row_is_reported():
    eng, srv = _setup()
    real = eng.prefill
    calls = []

    def prefill(tokens, st, head="last"):
        # between the pieces: one latent row of the first sparse layer moves by one bf16 step
        if st.length > 0 and not calls:
            calls.append(1)
            st.lat_all[0, st.length - 1, 0] = torch.nextafter(st.lat_all[0, st.length - 1, 0].float(),
                                                              torch.tensor(1e9)).to(st.lat_all.dtype) * 2
        return real(tokens, st, head=head)
    eng.prefill = prefill
    try:
        cases = RS.run(eng, srv, {"p": _ids(40)}, ["v2"], ["prefill"], lambda n: [20], 4, lambda line: None)
    finally:
        del eng.prefill
    c = cases["p.v2.prefill.k20"]
    assert calls and not c["equal"] and "lat_all" in c["state_differing"] and c["first_lat_layer"] == 0, c
    return f"reported: {c['state_differing']}, first layer {c['first_lat_layer']}, logits max |d| {c['logits_max_abs']:.3g}"


def test_the_k_list():
    ks = RS.ks_auto(423)
    assert {349, 390} <= set(ks) and {1, 31, 32, 33, 127, 128, 129, 422} <= set(ks), ks
    big = RS.ks_auto(4500)
    assert {4095, 4096, 4097, 4499, 4371, 4372} <= set(big) and all(1 <= k < 4500 for k in big)
    assert RS.piece_class(74, 349, "prefill", until=128) == "skinny" and RS.piece_class(33, 390, "prefill", until=128) == "skinny"
    assert RS.piece_class(4500, 0, "prefill", until=128) == "slab x1, last chunk slab"
    return f"n 423: {ks}"


def test_the_decode_tail_arm():
    eng, srv = _setup()
    ids = _ids(60, 9)
    got = []
    for mode in ("steps", "chain"):
        for k in (33, 47):
            line = RS.decode_tail(srv, ids, k, mode, 8, chunk=16)
            a, b = line["a"], line["b"]
            assert a["exact_ok"] and a["layer0_ok"], (mode, k, a)
            assert b["equal"], (mode, k, b)
            assert line["ok"], line
            got.append(f"{mode} k{k}: (a) {'equal' if a['equal'] else 'unequal'}, logits {a['logits_max_abs']:.3g}")
    return "; ".join(got) + "; (b) equal in every case"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
