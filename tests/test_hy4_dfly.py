"""DFly2 on a tiny random model, on the CPU: the draft head's arithmetic against a plain restatement of the
serving implementation it follows (expanded keys and values, explicit masks), the rows it reads from the
target (its latents at the last committed row's picks), and that a draft never changes the text."""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.hy4_flash.dfly import DFlyHead  # noqa: E402
from engine.models.hy4_flash.engine import Hy4FlashEngine  # noqa: E402
from engine.models.hy4_flash.ref import rotate  # noqa: E402
from engine.models.hy4_flash.serving import Hy4FlashServing  # noqa: E402
from tests.hy4_tiny import TINY, random_weights  # noqa: E402
from engine.models.hy4_flash.config import config_from_dict  # noqa: E402

TOP = dict(dfly_num_hidden_layers=2, dfly_mask_token_id=9, dfly_target_index_layer=[1, 5], dfly_target_kv_layers=[1, 5],
           dfly_target_index_topk=8, dfly_mtp_fusion="slot0", dfly_enable_hidden_correction=True,
           speculative_config={"method": "dfly", "num_speculative_tokens": 3})
_S = {}


def dfly_weights(cfg, seed=5) -> dict:
    g = torch.Generator().manual_seed(seed)

    def m(*shape):
        return torch.randn(*shape, generator=g) * (0.3 if len(shape) == 1 else 0.2)
    H, R, qr, a, I = cfg.hidden_size, cfg.kv_lora_rank, cfg.q_lora_rank, cfg.sparse_attn, 48
    W = {"dfly_layers.final_norm.weight": m(H), "dfly_layers.mtp_fusion.eh_proj.weight": m(H, 2 * H),
         "dfly_layers.mtp_fusion.enorm.weight": m(H), "dfly_layers.mtp_fusion.hnorm.weight": m(H), "dfly_layers.mtp_fusion.gate": m(H),
         "dfly_layers.hidden_correction.gate_proj.weight": m(H, 2 * H), "dfly_layers.hidden_correction.up_proj.weight": m(H, 2 * H),
         "dfly_layers.hidden_correction.down_proj.weight": m(H, H), "dfly_layers.hidden_correction.hidden_norm.weight": m(H),
         "dfly_layers.hidden_correction.embed_norm.weight": m(H)}
    for i in range(cfg.dfly.layers):
        p, s = f"dfly_layers.{i}.", f"dfly_layers.{i}.self_attn."
        W[p + "input_layernorm.weight"], W[p + "post_attention_layernorm.weight"] = m(H), m(H)
        W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"], W[p + "mlp.down_proj.weight"] = m(I, H), m(I, H), m(H, I)
        W[s + "linear_gate.weight"], W[s + "q_a_proj.weight"], W[s + "q_a_layernorm.weight"] = m(a.heads * a.v, H), m(qr, H), m(qr)
        W[s + "q_b_proj.weight"], W[s + "kv_a_proj_with_mqa.weight"] = m(a.heads * a.qk, qr), m(R + a.rope, H)
        W[s + "kv_a_layernorm.weight"], W[s + "kv_b_proj.weight"] = m(R), m(a.heads * (a.nope + a.v), R)
        W[s + "o_proj.weight"], W[s + "post_proj_norm.weight"], W[s + "learnable_sink_param"] = m(H, a.heads * a.v), m(H), m(a.heads)
    return W


def setup(dtype=torch.float32):
    if dtype not in _S:
        cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
        W, tables = random_weights(cfg)
        W.update(dfly_weights(cfg))
        _S[dtype] = (cfg, Hy4FlashEngine(cfg, W, tables, dtype=dtype, max_len=256))
    cfg, eng = _S[dtype]
    srv = Hy4FlashServing(eng, 256)
    return cfg, eng, srv, DFlyHead(srv)


def tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 60, (n,), generator=g).tolist()


def plain_hidden(cfg, eng, head, anchor, target_hidden, st):
    """The draft layers as the serving implementation states them: every key and value expanded from its
    latent, one mask over [context rows, block rows], the sink as one more logit."""
    W, d, a, R, H = eng.W, cfg.dfly, cfg.sparse_attn, cfg.kv_lora_rank, cfg.hidden_size
    Q, L = d.draft_tokens + 1, st.length

    def rms(x, w):
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)) * w

    e = W["embed_tokens.weight"][torch.tensor([anchor] + [d.mask_token_id] * d.draft_tokens)]
    f = torch.cat((rms(e[:1], W["dfly_layers.mtp_fusion.enorm.weight"]), rms(target_hidden, W["dfly_layers.mtp_fusion.hnorm.weight"])), -1)
    h = torch.cat((e[:1] + W["dfly_layers.mtp_fusion.gate"] * (f @ W["dfly_layers.mtp_fusion.eh_proj.weight"].T), e[1:]))
    cos, sin = eng.rope(eng.pos3(torch.arange(L, L + Q)), "sparse")
    for i in range(d.layers):
        p, s = f"dfly_layers.{i}.", f"dfly_layers.{i}.self_attn."
        x = rms(h, W[p + "input_layernorm.weight"])
        qres = rms(x @ W[s + "q_a_proj.weight"].T, W[s + "q_a_layernorm.weight"])
        q = (qres @ W[s + "q_b_proj.weight"].T).view(Q, a.heads, a.qk)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), -1)
        ckv = x @ W[s + "kv_a_proj_with_mqa.weight"].T
        own = torch.cat((rms(ckv[:, :R], W[s + "kv_a_layernorm.weight"]), rotate(ckv[:, R:], cos, sin)), -1)
        # the context: the target layer's cached rows at its picks for the last committed row
        pq, ok = st.picks[d.index_layers[i]]
        rows = [int(x_) for x_, k in zip(pq.tolist(), ok.tolist()) if k and x_ < L]
        lat = torch.cat((st.lat[d.kv_layers[i]][torch.tensor(rows, dtype=torch.int64)].float(), own))
        kv = (lat[:, :R] @ W[s + "kv_b_proj.weight"].T).view(-1, a.heads, a.nope + a.v)
        keys = torch.cat((kv[..., :a.nope], lat[:, None, R:].expand(-1, a.heads, -1)), -1)
        vals = kv[..., a.nope:]
        sc = torch.einsum("qhd,nhd->hqn", q, keys) * a.qk ** -0.5
        mask = torch.cat((torch.ones(Q, len(rows), dtype=torch.bool), torch.ones(Q, Q, dtype=torch.bool).tril()), -1)
        sc = sc.masked_fill(~mask[None], float("-inf"))
        sc = torch.cat((sc, W[s + "learnable_sink_param"].view(-1, 1, 1).expand(-1, Q, 1)), -1)
        pr = torch.softmax(sc, -1)[..., :-1]
        y = torch.einsum("hqn,nhv->qhv", pr, vals)
        y = (y * torch.sigmoid(x @ W[s + "linear_gate.weight"].T).view(Q, a.heads, a.v)).reshape(Q, -1)
        h = h + rms(y @ W[s + "o_proj.weight"].T, W[s + "post_proj_norm.weight"])
        x = rms(h, W[p + "post_attention_layernorm.weight"])
        h = h + (F.silu(x @ W[p + "mlp.gate_proj.weight"].T) * (x @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    return rms(h, W["dfly_layers.final_norm.weight"]), rows


def test_the_draft_layers_match_their_plain_form():
    cfg, eng, srv, head = setup()
    ids = tokens(45, 1)
    errs, seen = [], []
    for n in (6, 30):                               # before the indexer selects, and after
        srv.kv.length = 0
        lg = srv.forward(torch.tensor(ids[:n]), start=0)
        anchor = int(lg[0, 0].argmax())
        th = srv.last_streams()
        got = head.hidden(anchor, th, srv.st)
        want, rows = plain_hidden(cfg, eng, head, anchor, th, srv.st)
        errs.append((got - want).abs().max().item())
        seen.append(len(rows))
    assert max(errs) < 2e-5, errs
    assert seen == [6, 8], seen                     # every row so far; then 3 groups of 2 and the row's own
    return f"2 draft layers, a block of 4: {errs[0]:.1e} over {seen[0]} context rows, {errs[1]:.1e} over {seen[1]} picked rows"


def test_the_context_is_the_last_rows_picks_on_every_path():
    """After a prompt, after a decode step and after a committed block the head reads the rows the target's
    layer read for the last committed row, and the same draft comes out of each."""
    cfg, eng, srv, head = setup()
    ids = tokens(40, 2)
    a, b, c = Hy4FlashServing(eng, 256), Hy4FlashServing(eng, 256), Hy4FlashServing(eng, 256)
    a.forward(torch.tensor(ids[:31]), start=0)
    b.forward(torch.tensor(ids[:30]), start=0)
    b.forward(torch.tensor([ids[30]]), start=30)
    c.forward(torch.tensor(ids[:28]), start=0)
    c.forward_tree(torch.tensor([ids[28], 7, ids[29], ids[30]]), [-1, 0, 0, 2], start=28)
    c.commit_tree([0, 2, 3])
    picks = []
    for s in (a, b, c):
        assert s.st.picks_at == s.st.length == 31
        picks.append({li: sorted(q for q, k in zip(*(t.tolist() for t in s.st.picks[li])) if k) for li in (1, 5)})
    # row 30 opens a group: 3 selected groups of 2 and the row itself
    assert picks[0] == picks[1] == picks[2] and all(len(v) == 7 and v[-1] == 30 for v in picks[0].values()), picks
    outs = [DFlyHead(s).draft(11, s.last_streams(), s.st) for s in (a, b, c)]
    assert outs[1][0] == outs[2][0], outs                    # the two row paths hold the same bits
    assert all(len(o[0]) == 3 and len(o[0][0]) == 2 for o in outs)
    return f"picks {picks[0][5]} in layer 5 after a prompt, a step and a tree; drafts {outs[1][0]}"


def test_the_head_interface_and_a_stale_round():
    cfg, eng, srv, head = setup()
    assert head.depth == 3 and head.views() == []
    ids = tokens(20, 3)
    lg = srv.forward(torch.tensor(ids), start=0)
    head.reset()
    head.prime(torch.zeros(19, cfg.hidden_size), ids[1:])
    head.launch(srv.last_streams(), [int(lg[0, 0].argmax())])
    tok, prob = head.read()
    assert head.st.length == 20 and len(tok) == 3 and all(p[0] >= p[1] > 0 for p in prob) and head.read() == ([], [])
    # a length set by hand (a cache restore) leaves the picks stale: the round drafts nothing
    srv.kv.length = 12
    head.launch(srv.last_streams(), [5])
    assert head.read() == ([], []) and head.stats["stale"] == 1
    return f"3 drafted positions with their two best tokens; a stale round drafts nothing"


def test_speculative_greedy_is_plain_greedy():
    """Rounds of draft, verify and commit write the plain greedy text, token for token and logit for logit."""
    for dtype in (torch.float32, torch.bfloat16):
        cfg, eng, srv, head = setup(dtype)
        ids = tokens(22, 4)
        plain = Hy4FlashServing(eng, 256)
        lg = plain.forward(torch.tensor(ids), start=0)
        want, rows = [], []
        for i in range(24):
            t = int(lg[0, 0].argmax())
            want.append(t)
            lg = plain.forward(torch.tensor([t]), start=22 + i)
            rows.append(lg[0, 0].clone())
        lg = srv.forward(torch.tensor(ids), start=0)
        nxt, out, kept, rounds = int(lg[0, 0].argmax()), [], 0, 0
        while len(out) < 24:
            head.launch(srv.last_streams(), [nxt])
            tok, _ = head.read()
            block = [nxt] + [t[0] for t in tok]
            lg = srv.forward_tree(torch.tensor(block), list(range(-1, len(block) - 1)), start=srv.kv.length)
            picks = srv.picks.tolist()
            n = 1
            while n < len(block) and block[n] == picks[n - 1]:
                n += 1
            for j in range(n):
                assert torch.equal(lg[j], rows[len(out) + j]) if len(out) + j < 24 else True
            srv.commit_tree(list(range(n)))
            out += block[:n]
            nxt, kept, rounds = picks[n - 1], kept + n - 1, rounds + 1
        assert out[:24] == want, (out, want)
    return f"24 tokens equal in fp32 and bf16; bf16: {rounds} rounds, {kept} drafted tokens kept (random weights)"


def test_the_served_loop_drafts_and_writes_the_plain_text():
    """The server's request loop over the family's engine with the DFly head, the lookup and the cut (the chain,
    then with second-choice leaves): the completion is the plain one's text, and rounds were drafted."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_forward_tree as T  # noqa: F401  (the CPU environment before the server imports)
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok, Req
    saved = dict(app.STATE)
    body = {"prompt": "the quick brown fox jumps over the lazy dog and runs", "max_tokens": 24, "stream": False, "temperature": 0}
    out, stats = {}, {}
    try:
        for tag in ("plain", "chain", "tree"):
            cfg, eng, srv, head = setup(torch.bfloat16)
            drafter = None
            if tag != "plain":
                ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16)
                drafter = MTPTreeDrafter(srv, head, StairCut(PriceModel(), siblings=tag == "tree"), ng)
                srv.heads["dfly"] = head
            app.STATE.clear()
            app.STATE.update(engine=srv, drafter=drafter, k=head.depth, tree=drafter is not None, sampled_tree="det",
                             relax=Relax(1.0, 1), verbose=False, state_store=None, prefix_cache=False,
                             prefix_chunk=0, session_cache=False, tok=FakeTok(), device="cpu",
                             max_len=srv.max_len, default_max_tokens=64, pen_spec=PenaltySpec(), model="t",
                             cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                             request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None,
                             response_cache=None, suffix_store=None)
            texts = []
            for _ in range(2):
                headline, raw = Req("/v1/completions", body).response()
                assert headline.startswith("HTTP/1.1 200"), (tag, headline, raw[:400])
                r = json.loads(raw)
                assert r["usage"]["completion_tokens"] == 24, (tag, r["usage"])
                texts.append(r["choices"][0]["text"])
            assert texts[0] == texts[1], (tag, texts)
            out[tag], stats[tag] = texts[0], dict(srv.stats)
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    assert out["chain"] == out["plain"] and out["tree"] == out["plain"], out
    assert stats["chain"]["verify_blocks"] > 0 and stats["tree"]["verify_blocks"] > 0, stats
    return f"24 tokens twice, plain, chain and tree: the same text; {stats['chain']} / {stats['tree']}"


def test_the_drafter_line_counts_stale_rounds():
    """SRV-907: the verbose server's `[drafter]` line (the shared drafter's `report`) ends with the DFly head's
    count of stale rounds, which the proof runs read; a head without the count leaves the line as it was."""
    import types

    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut
    cfg, eng, srv, head = setup()
    d = MTPTreeDrafter(srv, head, StairCut(PriceModel(), siblings=False), NgramDrafter(corpus_path="", min_order=3,
                                                                                      max_depth=16))
    assert d.report().endswith(" stale=0"), d.report()[-40:]
    head.stats["stale"] += 2
    assert d.report().endswith(" stale=2")
    d.head = types.SimpleNamespace(stats={"rounds": 0}, depth=head.depth)
    assert " stale=" not in d.report()
    return "stale=0, then stale=2; none for a head without the count"


def test_the_first_round_after_each_resume_drafts():
    """SRV-907 (the conversation cache's design, test 10 and question 6.2.3): after a resume from each cache tier
    (an anchor, the end of a generation, the last prompt's end, a session snapshot, a prefix snapshot) the prefill
    forwards at least one row, which reads the picks again: the first round drafts, and `stale` stays 0."""
    from engine import cache
    cfg, eng, srv, _ = setup()
    ch, big = 16, 1 << 30
    res = cache.ResidentPrefix(big, ch, stash_bytes=big, end=True)
    store = cache.StateStore(big, chunk=ch)
    got = {}

    def turn(ids, k):
        info: dict = {}
        lg, _, _ = cache.prefill(srv, None, list(ids), "cpu", chunk=ch, resident=res, store=store, checkpoint=True,
                                 info=info, conv_id="A")
        head = DFlyHead(srv)
        head.reset()
        t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        head.launch(srv.last_streams(), [t])
        tok, _ = head.read()
        got[info.get("kind")] = (len(tok), head.stats["stale"])
        out, pos = [], len(ids)
        for i in range(k):
            out.append(t)
            if i == k - 1:
                break
            lg = srv.forward(torch.tensor([t]), start=pos)
            pos += 1
            t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        ctx = list(ids) + out
        store.put(ctx, cache.capture(srv), "A", kind="session")
        res.keep_end(srv, None, ctx, len(ids), [])
        return ctx

    keep, cache.PROMPT_END = cache.PROMPT_END, True
    try:
        p1 = tokens(40, 7)
        c1 = turn(p1, 20)
        p2 = c1 + tokens(5, 8)                                   # the end of turn 1
        c2 = turn(p2, 6)
        turn(p2 + [5 if c2[len(p2)] != 5 else 6] + tokens(8, 9), 4)    # turn 2's prompt end
        turn(p1[:35] + [5 if p1[35] != 5 else 6] + tokens(10, 10), 3)  # turn 1's anchor at 32
        res.clear()
        turn(c1 + tokens(3, 11), 3)                              # turn 1's session snapshot
        res.clear()
        turn(p1[:20] + [5 if p1[20] != 5 else 6] + tokens(4, 12), 2)    # turn 1's prefix snapshot at 16
    finally:
        cache.PROMPT_END = keep
    assert {"end", "pend", "resident", "session", "prefix"} <= set(got), got
    assert all(v == (3, 0) for k, v in got.items() if k is not None), got
    return f"drafted positions and stale rounds by resume: {got}"


def test_the_end_read_with_the_draft_head_gives_a_cold_reads_state():
    """The end read again (`Capabilities.reread_end`) with the served drafter (DFly through the tree drafter): turn 1
    drafts its answer through the server's loop, and its close reads the answer again with the drafter's own calls
    (`state_resume`, `sync`) and leaves the drafter where a resume at the end puts it: its head valid to the end's last
    row, so the session snapshot's drafter part covers the answer. Turn 2, resumed at that end and, the rows cleared,
    from the session snapshot, holds a cold read's target state (the logits, every section and view, bit for bit) and
    its answer through the drafted loop is the cold server's. The head's own rows are not held to bits: the head is approximate by
    design (plain turns skip rows, the backlog is trimmed; its drafts are verified). bf16: the served dtype, whose
    prompt pass gives the same bits at every cut on the CPU."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_forward_tree as T  # noqa: F401  (the CPU environment before the server imports)
    from engine import cache
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok
    ch, big = 16, 1 << 30

    def served():
        cfg, eng, srv, head = setup(torch.bfloat16)
        d = MTPTreeDrafter(srv, head, StairCut(PriceModel(), siblings=True),
                           NgramDrafter(corpus_path="", min_order=3, max_depth=16))
        srv.heads["dfly"] = head
        return srv, d

    def state(srv, res, store, d):
        app.STATE.clear()
        app.STATE.update(engine=srv, drafter=d, k=d.head.depth, tree=True, sampled_tree="det", relax=Relax(1.0, 1),
                         verbose=False, state_store=store, prefix_cache=True, prefix_chunk=ch, session_cache=True,
                         tok=FakeTok(), device="cpu", max_len=srv.max_len, default_max_tokens=64, pen_spec=PenaltySpec(),
                         model="t", cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                         request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None, response_cache=None,
                         suffix_store=None, resident=res)

    def prefill(srv, d, ids, **kw):
        d.reset()
        d.prime(list(ids))
        info: dict = {}
        lg, reused, _ = cache.prefill(srv, d, list(ids), "cpu", chunk=ch, info=info, **kw)
        return lg, info.get("kind"), reused

    def same(a, b, n):
        for (k, x), y in zip(a.cache_state().sections().items(), b.cache_state().sections().values()):
            assert torch.equal(x, y), k
        for (k, x, ax), (_, y, _) in zip(a.cache_state().views(), b.cache_state().views()):
            assert torch.equal(x.narrow(ax, 0, n), y.narrow(ax, 0, n)), k

    def turn1():
        srv, d = served()
        res = cache.ResidentPrefix(big, ch, stash_bytes=big, end=True, end_reread=True)
        store = cache.StateStore(big, chunk=ch)
        state(srv, res, store, d)
        p1 = tokens(37, 30)
        out1 = list(app.generate_stream(torch.tensor(p1), 24, set(), None, "A"))
        ctx1 = list(app.STATE["last_ctx"])
        assert srv.stats["verify_blocks"] > 0, srv.stats
        app._remember(list(p1), out1, "A")
        assert res.end is not None and res.end.get("exact") is True and res.stats.get("end_rereads") == 1, res.end
        e = res.end["at"]
        snap = [x for x in store._d.values() if x.kind == "session"]
        assert len(snap) == 1 and snap[0].exact and snap[0].snap.drafter["len"] == e - 1, \
            [(x.kind, x.exact, (x.snap.drafter or {}).get("len")) for x in store._d.values()]
        return srv, d, res, store, ctx1 + tokens(5, 31), e

    saved = dict(app.STATE)
    keep_pe, cache.PROMPT_END = cache.PROMPT_END, True
    got = []
    try:
        with torch.no_grad():
            for via in ("end", "session"):
                # the target's state after the resume, bit for bit against a cold read with a fresh drafter
                srv, d, res, store, p2, e = turn1()
                if via == "session":
                    res.clear()
                lg, kind, reused = prefill(srv, d, p2, resident=res, store=store, conv_id="A")
                assert kind == via and reused == e, (kind, reused, e)
                cs, cd = served()
                lg_c, _, _ = prefill(cs, cd, p2)
                assert torch.equal(lg, lg_c), (lg - lg_c).abs().max()
                same(srv, cs, len(p2))
                # the drafted answer from that resume, against a cold server's with a fresh drafter
                srv, d, res, store, p2, e = turn1()
                if via == "session":
                    res.clear()
                blocks = srv.stats["verify_blocks"]
                out2 = list(app.generate_stream(torch.tensor(p2), 12, set(), None, "A"))
                last = dict(app.STATE["last_prefill"])
                assert last["kind"] == via and last["reused"] == e, (last, srv.stats)
                drafted = srv.stats["verify_blocks"] - blocks       # the cut decides by its timings: recorded
                cs, cd = served()
                state(cs, None, None, cd)
                app.STATE.update(prefix_cache=False, prefix_chunk=0, session_cache=False)
                cold2 = list(app.generate_stream(torch.tensor(p2), 12, set(), None, "B"))
                assert out2 == cold2, (out2, cold2)
                got.append(f"{via} at {e}: the target's state and 12 tokens equal a cold read's ({drafted} drafted rounds)")
    finally:
        cache.PROMPT_END = keep_pe
        app.STATE.clear()
        app.STATE.update(saved)
    return "; ".join(got)


def test_drafted_is_plain_after_the_v2_prompt_pass():
    """SPD-323, SPD-325: after a prompt read on the prompt pass on fixed-order kernels (HY4_PREFILL_V2) drafted rounds
    write the plain greedy text, token for token and logit for logit, in fp32 and bf16, and through the served loop."""
    engs = [setup(d)[1] for d in (torch.float32, torch.bfloat16)]
    for e in engs:
        e.prefill_v2 = True
    try:
        return test_speculative_greedy_is_plain_greedy() + "; " + test_the_served_loop_drafts_and_writes_the_plain_text()
    finally:
        for e in engs:
            e.prefill_v2 = False


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
