"""ENG-476: tools/fn_server_check.py `--corners`, one run's answers against what the API says, on stand-in runs: a
run that keeps every corner passes; each broken corner fails by name (a stop string in the answer, a top list out of
order, a seeded request that differs when sent again, a forced call that names another tool or is text, a JSON answer
that is not JSON). The requests themselves are the ones the check sent to Flash-Next, LFM2.5 and NemotronH, unchanged.

CPU. Run: python tests/test_fn_server_check.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tools import fn_server_check as F  # noqa: E402


def _msg(content="", calls=None, finish="stop", logprobs=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                           for i, (n, a) in enumerate(calls)]
    ch = {"message": m, "finish_reason": finish}
    if logprobs is not None:
        ch["logprobs"] = {"content": logprobs}
    return {"choices": [ch]}


def _top(*lps):
    return [{"token": f"t{i}", "logprob": x} for i, x in enumerate(lps)]


def _run() -> dict:
    ans = {name: _msg("fine") for name in F.CASES}
    ans["stop_string"] = _msg("one, two, three, four, five, six, ")
    ans["logprobs"] = _msg("One two", logprobs=[{"token": "One", "logprob": -0.1, "top_logprobs": _top(-0.1, -2, -3, -4, -5)},
                                                {"token": " two", "logprob": 0.0, "top_logprobs": _top(0.0, -9, -9, -10, -11)}])
    for name in F.REPEAT:
        ans[name] = dict(_msg("a poem"), _again=_msg("a poem"))
    for name, body in F.CASES.items():
        tc = body.get("tool_choice")
        if tc:
            ans[name] = _msg(calls=[(tc["function"]["name"] if isinstance(tc, dict) else "get_weather", {"city": "Oslo"})],
                             finish="tool_calls")
    for name in ("json_format", "json_format_2", "json_format_3"):
        ans[name] = _msg('{"name": "Ada", "age": 36}')
    ans["stream_thinking"] = {"stream": True, "content": "Rhine", "reasoning_content": "x", "finish_reason": "stop"}
    return {"answers": ans}


def test_a_run_that_keeps_every_corner_passes():
    r = F.corners(_run())
    assert r["all_pass"], {k: v for k, v in r["cases"].items() if not v["pass"]}
    want = {"stop_string", "logprobs", "sampled_seed_repeats", "sampled_seed_code_repeats", "penalty_bias",
            "penalty_bias_2", "penalty_bias_3", "tool_forced", "tool_forced_2", "tool_forced_3", "json_format",
            "json_format_2", "json_format_3"}
    assert set(r["cases"]) == want, sorted(r["cases"])
    return f"{len(r['cases'])} corners"


def test_each_broken_corner_fails_by_name():
    cases = []

    def broken(name, fix):
        run = copy.deepcopy(_run())
        fix(run["answers"])
        r = F.corners(run)
        bad = sorted(k for k, v in r["cases"].items() if not v["pass"])
        assert not r["all_pass"] and bad == [name], (name, bad)
        cases.append(name)

    broken("stop_string", lambda a: a.update(stop_string=_msg("one, two, seven", finish="length")))
    broken("logprobs", lambda a: a["logprobs"]["choices"][0]["logprobs"]["content"][1].update(
        top_logprobs=_top(-9, 0.0, -9, -10, -11)))
    broken("sampled_seed_repeats", lambda a: a["sampled_seed"].update(_again=_msg("another poem")))
    broken("tool_forced", lambda a: a.update(tool_forced=_msg(calls=[("get_time", {})], finish="tool_calls")))
    broken("tool_forced_3", lambda a: a.update(tool_forced_3=_msg("<tool_call>get_weather</tool_call>")))
    broken("tool_forced_2", lambda a: a["tool_forced_2"]["choices"][0]["message"]["tool_calls"][0]["function"].update(
        arguments='{"city": "Oslo"'))
    broken("json_format_2", lambda a: a.update(json_format_2=_msg("Here you go: apple, red")))
    broken("penalty_bias", lambda a: a.update(penalty_bias=_msg("", finish="error")))
    assert not F.corners({"answers": {}})["all_pass"], "a run with no answers is no pass"
    return f"{len(cases)} broken corners, each failing by name"


def _written(token):
    """A token the server wrote (a budget's forced close), as server/logprobs.py reports one: logprob 0, itself the
    only alternative."""
    return {"token": token, "logprob": 0.0, "top_logprobs": [{"token": token, "logprob": 0.0}]}


def test_a_close_the_server_wrote_is_logprob_0_with_itself_alone():
    """ENG-476 (Hy4-flash, hold 1a on 5c89764b): the logprobs request's reasoning block is closed by the budget (the
    family's default cap less the answer reserve at 60 tokens), so its close is the server's: logprob 0 with itself
    the only alternative, every family alike (one token for Hy4-flash's own close, the shared phrase's tokens for
    the others). The corner takes one such run and still wants a full ordered top 5 for every token the model chose;
    a lone alternative that is not the token, a written token whose logprob is not 0, or a second run fail it."""
    def answer(entries):
        run = copy.deepcopy(_run())
        run["answers"]["logprobs"] = _msg("One two", logprobs=entries)
        return F.corners(run)["cases"]["logprobs"]
    first = {"token": "One", "logprob": -0.1, "top_logprobs": _top(-0.1, -2, -3, -4, -5)}
    after = {"token": " two", "logprob": 0.0, "top_logprobs": _top(0.0, -9, -9, -10, -11)}
    for close in (["</think:opensource>"], ["\n\n", "Considering", " the", "</think>", "\n\n"]):
        got = answer([first] + [_written(t) for t in close] + [after])
        assert got["pass"] and f"{len(close)} written" in got["seen"], (close, got)
    for label, entries in (
            ("a lone alternative that is not the token", [first, dict(_written("</think>"), top_logprobs=[
                {"token": "x", "logprob": 0.0}]), after]),
            ("a written token not at logprob 0", [first, dict(_written("</think>"), logprob=-0.3), after]),
            ("two written runs", [first, _written("</think>"), after, _written("</think>"), after])):
        got = answer(entries)
        assert not got["pass"], (label, got)
    return "one written run (one token, five) passes; a lone other token, a written -0.3, two runs fail"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
