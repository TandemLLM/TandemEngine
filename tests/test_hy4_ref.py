"""The Hy4-flash reference forward against the published implementation.

A tiny random Hy4-flash (6 layers: sparse, sparse, three window layers, sparse; layer 0 dense, the others a
latent MoE of 8 experts top 3 with a clamp low enough to bite; 4 residual streams; Engram on layers 1
and 3 with a lookup table that merges ids; a window of 4 tokens; an indexer of 2-token groups that
reads 8 tokens, so a 26-token prompt runs the window, the group pooling and the selection) is built
with `transformers`' `HYV4FlashTextModel` in fp32 with random weights, the zero-initialised ones
included; the engine's reference (`engine/models/hy4_flash/ref.py`) runs on the same weights.

The published code is pull request 50903 of `transformers` at b865850d. Where it is not installed the
comparison tests are skipped and say so; HY4_TRANSFORMERS may hold a path list (os.pathsep) with that
tree's `src` and the libraries it needs.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in reversed([p for p in os.environ.get("HY4_TRANSFORMERS", "").split(os.pathsep) if p]):
    sys.path.insert(0, _p)

from engine.models.flashnext.ple import DenseTable  # noqa: E402
from engine.models.hy4_flash.config import config_from_dict  # noqa: E402
from engine.models.hy4_flash.engram import EngramHash, fp8_round_trip, ngram_tokens  # noqa: E402
from engine.models.hy4_flash.ref import ExpertBank, Hy4FlashRef, Hy4State  # noqa: E402

TINY = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=6,
    num_attention_heads=4, swa_num_attention_heads=8, q_lora_rank=16, kv_lora_rank=8,
    qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_routed_experts=8, num_experts_per_tok=3,
    routed_expert_hidden_size=16, routed_scaling_factor=3.998, swiglu_limit=0.25, sliding_window=4,
    index_topk=8, index_head_dim=16, index_n_heads=2, compressed_indexer_compress_ratio=2, hc_mult=4,
    max_position_embeddings=256, rms_norm_eps=1e-5, pad_token_id=2, bos_token_id=1, eos_token_id=3,
    engram_layer_ids=[2, 4], engram_max_ngram_size=3, engram_n_head_per_ngram=2, engram_n_embed_per_ngram=8,
    engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]], "4": [[23, 29], [31, 37]]},
    engram_pad_id=2, engram_lookup_table_size=60,
    rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1],
                                           "mrope_interleaved": True},
                     "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1.0e7, "mrope_section": [2, 1, 1],
                                                   "mrope_interleaved": True}},
)
# the fp32 reference equals transformers within this bound (fp32 summation-order noise over 6 layers)
BOUND = 5e-6
g0 = torch.Generator().manual_seed(7)
# 26 tokens: ids 60..63 are past the lookup table (read as pad), id 2 is the pad token (owns no n-gram)
PROMPT = torch.cat([torch.tensor([5, 9, 3, 61, 22, 2, 3, 40, 41, 12, 63, 7]), torch.randint(4, 60, (14,), generator=g0)])


def _official():
    try:
        from transformers.models.hy_v4_flash import modeling_hy_v4_flash as m
        from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
    except Exception:
        return None
    torch.manual_seed(0)
    cfg = HYV4FlashTextConfig(**TINY)
    cfg._attn_implementation = "eager"
    model = m.HYV4FlashTextModel(cfg).float().eval()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("hc.scale", "hc_head.hc_scale")):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() == 1 else 0.2))
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.3)
        for mod in model.modules():
            if isinstance(mod, m.HYV4FlashEngramLayer):
                # a map that merges ids, as the checkpoint's does; the multipliers follow it, as on a load
                mod.lookup_table.copy_(torch.randint(0, 40, mod.lookup_table.shape, generator=g, dtype=torch.int32))
                mod.hash_multipliers.copy_(mod.draw_hash_multipliers(int(mod.lookup_table.max()) + 1))
    head = torch.randn(TINY["vocab_size"], TINY["hidden_size"], generator=g) * 0.2
    return model, head


_RENAME = (
    (".self_attn.sinks", ".self_attn.learnable_sink_param"), (".self_attn.gate_proj.weight", ".self_attn.linear_gate.weight"),
    (".post_mlp_layernorm.weight", ".mlp.post_fc2_norm.weight"), (".mlp.pre_expert_proj.weight", ".mlp.routed_expert_down_proj.weight"),
    (".mlp.post_expert_proj.weight", ".mlp.routed_expert_up_proj.weight"),
    (".attn_hc.fn", ".hc_attn_layer.hc_pre.hc_fn"), (".attn_hc.base", ".hc_attn_layer.hc_pre.hc_base"),
    (".attn_hc.scale", ".hc_attn_layer.hc_pre.hc_scale"), (".ffn_hc.fn", ".hc_mlp_layer.hc_pre.hc_fn"),
    (".ffn_hc.base", ".hc_mlp_layer.hc_pre.hc_base"), (".ffn_hc.scale", ".hc_mlp_layer.hc_pre.hc_scale"),
    ("hc_head.hc_fn", "hc_head.hc_head_fn"), ("hc_head.hc_base", "hc_head.hc_head_base"), ("hc_head.hc_scale", "hc_head.hc_head_scale"),
)


def weights_from_official(model, head) -> tuple[dict, dict]:
    """The published model's tensors under the checkpoint's names (the renames of its conversion mapping, reversed)."""
    W, tables = {"lm_head.weight": head.clone()}, {}
    for k, v in model.state_dict().items():
        name = k
        for new, old in _RENAME:
            if name.endswith(new):
                name = name[: -len(new)] + old
        if name.endswith("mlp.experts.gate_up_proj"):
            base, inter = name[: -len("gate_up_proj")], v.shape[1] // 2
            W[base + "gate_proj"] = ExpertBank(dense=v[:, :inter].clone())
            W[base + "up_proj"] = ExpertBank(dense=v[:, inter:].clone())
        elif name.endswith("mlp.experts.down_proj"):
            W[name] = ExpertBank(dense=v.clone())
        elif name.endswith("multi_head_embedding.embedding.weight"):
            tables[int(name.split(".")[1])] = DenseTable(v.clone())
        else:
            W[name] = v.clone()
    return W, tables


_CACHE = {}


def _pair():
    if "pair" not in _CACHE:
        off = _official()
        if off is None:
            _CACHE["pair"] = (None, None, None)
        else:
            model, head = off
            W, tables = weights_from_official(model, head)
            _CACHE["pair"] = (model, head, Hy4FlashRef(config_from_dict(dict(TINY)), W, tables, dtype=torch.float32))
    return _CACHE["pair"]


class _StableTopk:
    """`torch.topk` leaves the order of equal values open, and the indexer's scores tie often (a relu
    over few heads gives exact zeros). While the published model runs here, ties resolve to the lower
    index, the engine's rule: one of the results `topk` may return."""

    def __enter__(self):
        import collections
        self.fn, self.method = torch.topk, torch.Tensor.topk
        out = collections.namedtuple("topk", ["values", "indices"])

        def topk(x, k, dim=-1, largest=True, sorted=True):
            assert largest
            v, i = torch.sort(x, dim=dim, descending=True, stable=True)
            return out(v.narrow(dim, 0, k), i.narrow(dim, 0, k))
        torch.topk, torch.Tensor.topk = topk, topk
        return self

    def __exit__(self, *exc):
        torch.topk, torch.Tensor.topk = self.fn, self.method


def _official_run(model, head, tokens, hooks=None):
    """The published model on a whole prompt: fp32 logits [T, V] through the test's own head."""
    ids = tokens[None]
    grams = model.build_ngram_tokens(ids, None)
    mask = ids != TINY["engram_pad_id"]
    with torch.no_grad(), _StableTopk():
        out = model(input_ids=ids, ngram_tokens=grams, engram_mask=mask, use_cache=False)
    return out.last_hidden_state[0].float() @ head.T


SKIP = "skipped: transformers without hy_v4_flash (HY4_TRANSFORMERS)"


def test_logits_match_official():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    want = _official_run(model, head, PROMPT)
    got = ref.forward(PROMPT, Hy4State())
    err = (got - want).abs().max().item()
    assert err < BOUND, f"max abs err {err}"
    assert torch.equal(got.argmax(-1), want.argmax(-1))
    return f"26 tokens (window of 4, sparse past token 8), max abs err {err:.1e}"


def test_layer_by_layer_match_official():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    outs = []
    hooks = [layer.register_forward_hook(lambda m, i, o: outs.append(o[0] if isinstance(o, tuple) else o))
             for layer in model.layers]
    _official_run(model, head, PROMPT)
    for h in hooks:
        h.remove()
    trace = []
    ref.forward(PROMPT, Hy4State(), trace=trace)
    errs = [(a[0] - b).abs().max().item() for a, b in zip(outs, trace)]
    assert len(errs) == 6 and max(errs) < BOUND, errs
    return "per-layer max abs err " + " ".join(f"{e:.1e}" for e in errs)


def test_decode_matches_official_cache():
    """A prefix, then one token at a time: the reference with its state against the published model with
    its cache (the window layers' cropped rows, the indexer's cached token keys and positions)."""
    model, head, ref = _pair()
    if model is None:
        return SKIP
    from transformers import DynamicCache
    cut = 7
    ids = PROMPT[None]
    grams = model.build_ngram_tokens(ids, None)
    mask = ids != TINY["engram_pad_id"]
    cache = DynamicCache(config=model.config)
    want = []
    with torch.no_grad(), _StableTopk():
        o = model(input_ids=ids[:, :cut], ngram_tokens=grams[:, :cut], engram_mask=mask[:, :cut], cache=cache, use_cache=True)
        want.append(o.last_hidden_state[0].float() @ head.T)
        for t in range(cut, PROMPT.shape[0]):
            o = model(input_ids=ids[:, t:t + 1], ngram_tokens=grams[:, t:t + 1], engram_mask=mask[:, t:t + 1],
                      cache=o.cache, use_cache=True)
            want.append(o.last_hidden_state[0].float() @ head.T)
    want = torch.cat(want)
    st = Hy4State()
    got = [ref.forward(PROMPT[:cut], st)] + [ref.forward(PROMPT[t:t + 1], st) for t in range(cut, PROMPT.shape[0])]
    got = torch.cat(got)
    err = (got - want).abs().max().item()
    assert err < BOUND, f"max abs err {err}"
    whole = _official_run(model, head, PROMPT)
    drift = (want - whole).abs().max().item()
    return f"prefix 7 + 19 decode steps, max abs err {err:.1e} (the published cache path against its own whole prompt: {drift:.1e})"


def test_chunking_does_not_move_the_picks():
    """Whole prompt, chunks and single steps select the same rows: the logits agree within summation order."""
    model, head, ref = _pair()
    if model is None:
        return SKIP
    a = ref.forward(PROMPT, Hy4State())
    st = Hy4State()
    b = torch.cat([ref.forward(PROMPT[i:i + 5], st) for i in range(0, PROMPT.shape[0], 5)])
    err = (a - b).abs().max().item()
    assert err < BOUND, err
    return f"chunks of 5 against the whole prompt: {err:.1e}"


def test_sparse_window_engram_and_clamp_are_exercised():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    base = ref.forward(PROMPT, Hy4State())
    moved = {}
    for name, change in (("indexer", dict(index_topk=1 << 10)), ("window", dict(sliding_window=1 << 10)),
                         ("clamp", dict(swiglu_limit=1.0e6)), ("engram", dict(engram_fp8_embedding=False))):
        other = Hy4FlashRef(config_from_dict(dict(TINY, **change)), ref.W, ref.tables, dtype=torch.float32)
        moved[name] = (other.forward(PROMPT, Hy4State()) - base).abs().max().item()
    assert all(v > 1e-3 for k, v in moved.items() if k != "engram") and moved["engram"] > 1e-6, moved
    # the first rows are dense and inside the window: only later rows move
    wide = Hy4FlashRef(config_from_dict(dict(TINY, index_topk=1 << 10, sliding_window=1 << 10)), ref.W, ref.tables, dtype=torch.float32)
    d = (wide.forward(PROMPT, Hy4State()) - base).abs().amax(-1)
    assert d[:4].max().item() < BOUND and d[10:].max().item() > 1e-3, d.tolist()
    # zero-initialised parameters carry weight in this fixture
    for name, tensor in (("sinks", "layers.2.self_attn.learnable_sink_param"), ("router bias", "layers.1.mlp.gate.e_score_correction_bias")):
        W2 = dict(ref.W)
        W2[tensor] = torch.zeros_like(ref.W[tensor])
        other = Hy4FlashRef(ref.cfg, W2, ref.tables, dtype=torch.float32)
        moved[name] = (other.forward(PROMPT, Hy4State()) - base).abs().max().item()
        assert moved[name] > 1e-4, moved
    return "turned off, each moves the logits: " + ", ".join(f"{k} {v:.1e}" for k, v in moved.items())


def test_engram_ids_match_official():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(0, TINY["vocab_size"], (10000,), generator=g)
    grams_off = model.build_ngram_tokens(toks[None], None)[0]
    grams = ngram_tokens(toks, toks[:0], torch.arange(10000), 3, TINY["engram_pad_id"])
    assert torch.equal(grams, grams_off)
    n = 0
    for li, layer in enumerate(model.layers):
        if layer.engram_layer is None:
            continue
        e = layer.engram_layer
        want = e.hash_ngrams(grams_off) + e.multi_head_embedding.offsets
        assert torch.equal(ref.hashes[li].ids(grams), want), li
        n += want.numel()
    # a continuation reads its history: the same rows as the whole sequence
    tail = ngram_tokens(toks[100:], toks[:100], torch.arange(100, 10000), 3, TINY["engram_pad_id"])
    assert torch.equal(tail, grams[100:])
    return f"{n} row indices of 10,000 n-grams equal, two layers"


def test_engram_hash_at_full_size():
    """The real model's sizes: the multipliers' bound keeps every product inside int64, and the ids stay in range."""
    sizes = [10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147,
             10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259]
    lookup = torch.arange(120818, dtype=torch.int32)
    h = EngramHash(lookup, sizes, 3, 8, 120002, 2, 0)
    assert all(m % 2 == 1 and 0 < m * 120817 < 2 ** 63 for m in h.multipliers.tolist())
    g = torch.Generator().manual_seed(5)
    grams = torch.randint(0, 120832, (4096, 3), generator=g)
    ids = h.head_ids(grams)
    assert ids.shape == (4096, 16) and int(ids.min()) >= 0 and bool((ids < h.sizes).all())
    # python integers as the witness of the int64 arithmetic
    row = grams[17].tolist()
    c = [int(lookup[t]) if t < 120818 else 120002 for t in row]
    m = [c[j] * int(h.multipliers[j]) for j in range(3)]
    want = [(m[0] ^ m[1]) % s for s in sizes[:8]] + [(m[0] ^ m[1] ^ m[2]) % s for s in sizes[8:]]
    assert ids[17].tolist() == want
    assert h.rows == sum(sizes) and int(h.ids(grams).max()) < h.rows
    return f"16 heads, {h.rows:,} rows a table, ids in range"


def test_fp8_round_trip_is_the_reference():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import _fp8_round_trip
    g = torch.Generator().manual_seed(4)
    for dtype in (torch.float32, torch.bfloat16):
        x = (torch.randn(512, 96, generator=g) * 0.05).to(dtype)
        assert torch.equal(fp8_round_trip(x), _fp8_round_trip(x)), dtype
    return "equal bits in fp32 and bf16"


def test_state_truncate_resumes():
    model, head, ref = _pair()
    if model is None:
        return SKIP
    st = Hy4State()
    a = ref.forward(PROMPT, st)
    st.truncate(15)
    assert st.length == 15 and all(v.shape[0] == 15 for d in (st.kp, st.kr, st.ik) for v in d.values())
    b = ref.forward(PROMPT[15:], st)
    assert st.length == PROMPT.shape[0]
    err = (a[15:] - b).abs().max().item()
    assert err < BOUND, err
    return f"rows after a truncate to 15 equal the whole prompt's within {err:.1e}"


def test_bf16_runs_and_mostly_agrees():
    """The serving dtype: the fp32 islands (hyper-connections, router, indexer, sinks, head) and the bf16
    Engram round trip run, and the next token mostly agrees with fp32. A tiny random model flips a routed
    expert or a group pick on bf16 noise now and then, and such a row moves freely; the gates on real
    weights are the quality gate and the engine's own row checks."""
    model, head, ref = _pair()
    if model is None:
        return SKIP
    W, tables = weights_from_official(model, head)
    ref16 = Hy4FlashRef(ref.cfg, W, tables, dtype=torch.bfloat16)
    a = ref.forward(PROMPT, Hy4State())
    st = Hy4State()
    b = ref16.forward(PROMPT, st)
    assert b.dtype == torch.float32 and all(v.dtype == torch.bfloat16 for d in (st.kp, st.kr, st.ik) for v in d.values())
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    assert agree >= 0.75, agree
    return f"bf16 vs fp32 next-token agreement {agree:.2f} on 26 rows"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
