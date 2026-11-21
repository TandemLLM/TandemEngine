"""ENG-406 (ENG-E17): what a family tells the server about its model, as data on its registry entry.

  1  every registered family has a ServeSpec; the families that differ from the default say so in their
     own arch.py (the arg-key form and a model that opens its own reasoning block; Pythonic call lists)
  2  the server reads the served family's spec (`app._spec()`), the default when none was loaded
  3  thinking off: a model that opens its own block gets an empty block after the generation prompt; a
     template that opened the block gets it closed; anything else is the template's own prompt
  4  one default family (`default_arch`) serves a `--model` that names no checkpoint directory

CPU, the fake engine in this process. Run: python tests/test_modelspec.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.models import REGISTRY, default_arch  # noqa: E402
from engine.models.adapter import EffortSpec, ReasoningSpec, ServeSpec  # noqa: E402
from server import app, fake_engine  # noqa: E402
from server.stream import CLOSE_THINK, OPEN_THINK  # noqa: E402


def _serve(spec: ServeSpec | None) -> None:
    a = app.parser().parse_args(["--fake-engine", "--fake-tps", "200000", "--max-len", "65536"])
    app.STATE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(app, a)
    if spec is not None:
        app.STATE["spec"] = spec


def test_every_family_has_a_spec():
    for name, a in REGISTRY.items():
        assert isinstance(a.spec, ServeSpec) and isinstance(a.spec.reasoning, ReasoningSpec), name
        assert a.spec.tool_format and a.spec.tool_format == a.spec.tool_format.strip(), name
    own = {n: (a.spec.tool_format, a.spec.reasoning.opens_itself, a.spec.prompt.system_first, a.spec.side_slots)
           for n, a in REGISTRY.items() if a.spec != ServeSpec()}
    # SPD-217: Ornith (qwen3_5_moe) builds side slots now, so its spec is the default one
    # Hy4-flash builds side slots too (ENG-485: its lanes take turns)
    assert own == {"flint": ("argkey", True, False, True), "lfm2_moe": ("pythonic", False, True, True),
                   "hy_v4_flash": ("argkey", False, True, True)}, own
    return f"{len(REGISTRY)} families, {len(own)} with their own spec"


def test_the_default_effort_spec_is_the_shared_words():
    """SRV-899: the default EffortSpec is the shared template words, each its own word, and no default of its own;
    every family but Hy4-flash keeps it."""
    d = EffortSpec()
    assert tuple(dict(d.words)) == tuple(app.thinking_mod.TEMPLATE_WORDS), d.words
    assert all(lv == w for lv, w in d.words) and d.default is None
    assert ReasoningSpec().effort == d and ServeSpec().reasoning.effort == d
    for word in ("off", "low", "medium", "high", "xhigh", None, "odd"):
        want = word if word in app.thinking_mod.TEMPLATE_WORDS else None
        assert d.word(word) == want, word
    own = sorted(n for n, a in REGISTRY.items() if a.spec.reasoning.effort != d)
    assert own == ["hy_v4_flash"], own
    return f"{len(REGISTRY) - len(own)} families on the shared words, own: {own or 'none'}"


def test_the_spec_says_what_the_engine_builds():
    """#104: a family's ServeSpec.side_slots (what the Setup Guide reads) is its engine's
    Capabilities.side_slots (what the server reads at load)."""
    import importlib
    import inspect
    from engine.models.adapter import Capabilities
    seen = {}
    for name, a in REGISTRY.items():
        pkg = a.serve.split(":")[0].rsplit(".", 1)[0] if a.serve else "engine.models.qwen3_5"
        mod = importlib.import_module(pkg + ".serving")
        caps = [c.caps for _, c in inspect.getmembers(mod, inspect.isclass)
                if c.__module__ == mod.__name__ and isinstance(getattr(c, "caps", None), Capabilities)]
        if not caps and name == "qwen3_5":       # the 27B's engine carries its own (engine/models/qwen3_5/adapter.py)
            importlib.import_module("engine.models.qwen3_5.adapter")
            caps = [a.engine_class().caps]
        assert caps, (name, "no engine class with Capabilities")
        assert all(c.side_slots == a.spec.side_slots for c in caps), (name, a.spec.side_slots, caps)
        seen[name] = a.spec.side_slots
    # SPD-217: Ornith too (its joint verify pass); ENG-485: Hy4-flash too (its lanes take turns)
    assert all(seen.values()), seen
    return seen


def test_health_names_the_slot_it_runs():
    """#104: `--parallel 2` on an engine without side slots runs one slot, and /health says so."""
    _serve(None)
    assert app._par_status() == {}
    app.STATE["parallel_unbuilt"] = 2
    p = app._par_status()["parallel"]
    assert p["requested"] == 2 and p["max_running"] == 1 and p["n"] == 1 and "not built" in p["max_running_reason"]
    app.STATE.pop("parallel_unbuilt")


def test_the_server_reads_the_served_spec():
    _serve(None)
    assert app._spec() is app.DEFAULT_SPEC and app._spec() == ServeSpec()
    s = ServeSpec(tool_format="argkey", reasoning=ReasoningSpec(opens_itself=True))
    _serve(s)
    assert app._spec() is s


def _prompt(spec: ServeSpec | None, thinking: bool) -> str:
    _serve(spec)
    body = {"messages": [{"role": "user", "content": "Say hi."}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    client = app.thinking_mod.read(body, app._effort_caps())
    ids = app.build_prompt(body, client, count="ids")
    return app.STATE["tok"].decode(ids, skip_special_tokens=False)


def test_thinking_off_renders_by_the_spec():
    plain_on, plain_off = _prompt(None, True), _prompt(None, False)
    own = ServeSpec(reasoning=ReasoningSpec(opens_itself=True))
    own_on, own_off = _prompt(own, True), _prompt(own, False)
    assert own_on == plain_on, "thinking on: the spec changes nothing"
    # 0.14.0's render (SRV-685): a model that opens its own block gets an empty one after the generation
    # prompt, unless the template left a block open, which the server closes as for any model
    assert not plain_off.rstrip().endswith(OPEN_THINK), "the fake template closes its own block"
    assert own_off == plain_off + OPEN_THINK + CLOSE_THINK, own_off[-80:]
    return repr(own_off[-40:])


def _ask(spec, extra: dict, default="xhigh", think_budget=0, limit=32768):
    """One request through the handler with the fake engine and `spec` in force (tests/test_effort_table.py's
    harness, which puts the server back after it): the template call's arguments, the budget the loop got, the log.
    `extra` goes over the body last (`stream`, `max_tokens`)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import test_effort_table as table
    with table._served(spec, default) as seen:
        app.STATE["think_budget"] = think_budget
        tok = app.STATE["tok"]
        body = dict({"max_tokens": limit, "stream": False}, messages=[{"role": "user", "content": "Say hi."}], **extra)
        out = io.StringIO()
        req = table._Req(body)
        with contextlib.redirect_stdout(out):
            req.do_POST()
        # the close the reserve counts, read as the server reads it (`_think_close()`, under `spec`): the family's
        # own (`ReasoningSpec.close`, Hy4-flash's `</think>`), else the shared phrase; no monitor holds the cap here
        _ask.close = app._think_close()
        _ask.closer = len(tok(_ask.close, add_special_tokens=False).input_ids)
        kw = tok.calls[0]["kwargs"] if tok.calls else {}
    return kw, (seen[0]["budget"] if seen else None), out.getvalue()


def _one(spec, extra: dict, default="xhigh", think_budget=0, limit=32768):
    """`_ask`'s template word only."""
    kw, budget, log = _ask(spec, extra, default=default, think_budget=think_budget, limit=limit)
    _one.close, _one.closer = _ask.close, _ask.closer
    return kw.get("reasoning_effort"), budget, log


def test_a_family_default_renders_its_word_and_caps_what_nobody_decided():
    """SRV-899: a family whose spec has a default level of its own renders that level's word where the client sent
    none, in place of --reasoning-effort's, and caps the request at the level's cap with the answer reserve. An
    explicit word, a number, thinking off and --think-budget each leave the default cap out (the plan that decides
    the effort: tests/test_client_budgets.py)."""
    words = (("low", "w-low"), ("medium", "w-medium"), ("high", "w-high"), ("xhigh", "w-xhigh"))
    spec = ServeSpec(reasoning=ReasoningSpec(effort=EffortSpec(words=words, default="medium")))
    th = app.thinking_mod
    word, budget, log = _one(spec, {})
    assert word == "w-medium" and budget == 8192, (word, budget)
    assert "server default medium: cap 8192" in log and "server default: answer reserve" in log, log
    # a spec with no close of its own: the reserve counts the shared phrase
    assert _one.close == app.ThinkBudget.PHRASE, _one.close
    closer = _one.closer
    word, budget, log = _one(spec, {}, limit=4000)
    assert budget == th.with_reserve(8192, 4000, th.AUTO, closer)[0] < 8192, budget
    # an explicit word: its own word and its own cap (xhigh: none, then the client's reserve)
    word, budget, log = _one(spec, {"reasoning_effort": "xhigh"})
    assert word == "w-xhigh" and budget == th.with_reserve(0, 32768, th.AUTO, closer)[0], (word, budget)
    assert "server default" not in log and "client budget" in log
    word, budget, log = _one(spec, {"reasoning_effort": "high"})
    assert word == "w-high" and budget == 16384 and "server default" not in log, (word, budget)
    # a number: the default's word, the number's cap
    word, budget, log = _one(spec, {"max_reasoning_tokens": 512})
    assert word == "w-medium" and budget == 512 and "server default" not in log, (word, budget)
    # thinking off, by the client and by the server's default
    for extra, default in (({"chat_template_kwargs": {"enable_thinking": False}}, "xhigh"), ({}, "off")):
        word, budget, log = _one(spec, extra, default=default)
        assert not budget and "server default" not in log, (extra, default, budget)
    # --think-budget is the operator's: it wins
    word, budget, log = _one(spec, {}, think_budget=300)
    assert budget == 300 and "server default" not in log, budget
    # the default spec: no default of its own, the server's word, no cap of a level's: the answer reserve alone
    # (ENG-468 / M1.5, every request with thinking on keeps it)
    word, budget, log = _one(None, {})
    assert word == "xhigh" and "server default" not in log, (word, budget)
    assert budget == th.with_reserve(0, 32768, th.AUTO, _one.closer)[0] and "answer reserve" in log, (budget, log)


HY4 = REGISTRY["hy_v4_flash"].spec if "hy_v4_flash" in REGISTRY else None


def test_hy4_a_clients_xhigh_or_max_reaches_max():
    """SRV-899 on Hy4-flash's own spec: a client's own xhigh or max, top level or in chat_template_kwargs, gives the
    template's `max` with no cap but the client's answer reserve; a client's high gives its normal mode `high`."""
    th = app.thinking_mod
    for extra in ({"reasoning_effort": "xhigh"}, {"reasoning_effort": "max"},
                  {"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
                  {"chat_template_kwargs": {"reasoning_effort": "max"}}):
        kw, budget, log = _ask(HY4, extra)
        assert kw.get("reasoning_effort") == "max", (extra, kw)
        assert _ask.close == HY4.reasoning.close == "</think>", _ask.close   # the family's own close, counted
        assert budget == th.with_reserve(0, 32768, th.AUTO, _ask.closer)[0], (extra, budget)
        assert "server default" not in log and "client budget 0" in log, log
    kw, budget, _ = _ask(HY4, {"reasoning_effort": "high"})
    assert kw.get("reasoning_effort") == "high" and budget == 16384, (kw, budget)
    return "xhigh and max, top level and in kwargs: max"


def test_hy4_server_default_off_turns_thinking_off():
    """SRV-899: `--reasoning-effort off` on Hy4-flash: a request that says nothing is thinking off (the tokenizer
    renders `no_think`), with no cap and no default line; one that asks for thinking keeps it."""
    kw, budget, log = _ask(HY4, {}, default="off")
    assert kw.get("enable_thinking") is False and not budget and "server default" not in log, (kw, budget)
    kw, budget, log = _ask(HY4, {"reasoning_effort": "low"}, default="off")
    assert kw.get("enable_thinking") is True and kw.get("reasoning_effort") == "high" and budget == 1024, (kw, budget)
    return "off: no_think; a request's own low: high with 1,024"


def test_hy4_default_cap_streamed():
    """SRV-899 on Hy4-flash's spec: with no effort asked and nothing else deciding, a streamed request gets the
    family's default (the mode `high`, the medium cap, the reserve), as a non-streamed one does."""
    th = app.thinking_mod
    for stream in (False, True):
        for limit in (32768, 2048):
            kw, budget, log = _ask(HY4, {"stream": stream}, limit=limit)
            assert kw.get("reasoning_effort") == "high", (stream, kw)
            assert _ask.close == HY4.reasoning.close == "</think>", _ask.close
            assert budget == th.with_reserve(8192, limit, th.AUTO, _ask.closer)[0], (stream, limit, budget)
            assert "server default medium: cap 8192" in log and "server default: answer reserve" in log, log
    return "streamed and not, 32,768 and 2,048: the medium cap with the reserve"


def test_the_start_note_names_a_word_a_family_does_not_render():
    """SRV-899: a family with its own default says once at start that --reasoning-effort's word is not used (xhigh,
    low), and says nothing for its own level (medium), for off (no word left) and unset; a family without a default
    says nothing."""
    saved = dict(app.STATE)
    try:
        for spec, word, says in ((HY4, "xhigh", True), (HY4, "low", True), (HY4, "medium", False),
                                 (HY4, "off", False), (HY4, None, False), (None, "xhigh", False),
                                 (REGISTRY["lfm2_moe"].spec, "low", False), (REGISTRY["flint"].spec, "xhigh", False)):
            a = app.parser().parse_args(["--fake-engine"] + (["--reasoning-effort", word] if word else []))
            app._thinking_default(a)
            app.STATE.clear()
            app.STATE["spec"] = spec
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                app._effort_default_note(a)
            text = out.getvalue()
            assert ("is not used" in text) is says, (word, spec is HY4, text)
            if says:
                assert f"--reasoning-effort {word} is not used: this model's default is medium (cap 8192)" in text, text
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return "xhigh, low: the note; medium, off, unset, other families: none"


def test_thinking_off_adds_the_spec_line_before_the_conversation():
    """SRV-697: a spec with an off line has it before the conversation (after the client's system messages)
    with thinking off, and nowhere with thinking on; a spec without one adds nothing."""
    note = "Answer directly. Do not show your reasoning."
    spec = ServeSpec(reasoning=ReasoningSpec(opens_itself=True, off_note=note, off_role="system"))
    off, on = _prompt(spec, False), _prompt(spec, True)
    assert note in off and note not in on, off[-200:]
    assert off.index(note) < off.index("Say hi."), "the line comes before the user's message"
    assert note not in _prompt(None, False)
    flint = REGISTRY["flint"].spec.reasoning
    assert flint.off_note and flint.off_role == "developer"


def test_where_system_messages_may_stand():
    """A family whose template reads one system prompt first (the default, the 27B's): developer read as
    system, a leading run joined, one later refused. A family whose template takes them where they stand
    (Flint): kept in place with their role."""
    from server import compat
    msgs = [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
            {"role": "user", "content": "hi"}, {"role": "developer", "content": "C"}]
    try:
        compat.messages({"messages": msgs})
    except compat.Refusal as exc:
        assert "must come before the conversation" in str(exc.body())
    else:
        raise AssertionError("the default refuses a developer message after the conversation began")
    assert compat.messages({"messages": msgs[:3]}) == [{"role": "system", "content": "A\n\nB"},
                                                       {"role": "user", "content": "hi"}]
    assert compat.messages({"messages": msgs}, system_first=False) == msgs


def test_one_default_family():
    d = default_arch()
    assert sum(a.default for a in REGISTRY.values()) == 1 and app._arch_of(None) is d
    assert app._arch_of("/nonexistent/dir") is d


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
