"""SRV-111 / SRV-112: a request that offered tools never gets an empty stop.

The test matrix is SRV-111's Gherkin, on the served loop with the random 4-layer model and the
handler (the end token is whatever the model draws at a chosen step, so it can be scripted):

  1  the end token inside the reasoning block: the engine drops it, forwards the hidden close,
     the client receives `</think>` and the continued answer, never the phrase -- every format;
     a close the model writes as its first token (Hy4-flash, its tags, tokenizer and spec) ends
     the block the prompt opened: an answer, not an empty turn, no act (SRV-955)
  2  the end token after the reasoning with an empty answer: a hidden user turn with the nudge
     and an empty closed reasoning block; no second `<think>` on the wire. Both after the
     engine's own budget closed the block (the end token right after the streamed phrase) and
     with thinking off (the end token as the first token)
  3  two failures end in GIVE_UP, with the stronger nudge in between
  4  a text answer is left alone
  5  no tools, no action
  6  tool_choice none is respected
  7  shadow only records

and n > 1 and a stop string pass too. That the continued answer is a tool call opencode can run
is the fake engine's e2e (tests/test_fake_engine.py, `test_conductor_on_over_the_wire`): this
model writes noise.

Run: python tests/test_empty_turn.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import pairapp  # noqa: E402,F401  (a pair test: TandemApp from TANDEMAPP_SRC, ENG-413 WP14)
import torch  # noqa: E402

from engine.models.hy4_flash.arch import ARCH as HY4  # noqa: E402
from engine.models.hy4_flash.tokenizer import TAGS, WRAPPERS, Hy4Tokenizer  # noqa: E402
from engine.spec import ThinkBudget  # noqa: E402
from server import app  # noqa: E402
app.EXT = app.ext_mod.attach(app.STATE)   # ENG-413 WP3b: the request path asks the conductor and the
#                                       monitor this file builds through the extension
from test_app_loop import Req, serve  # noqa: E402
from tandemapp.conductor.monitor import Journal, Monitor  # noqa: E402
from test_supervisor_frame import (  # noqa: E402
    ALPHA, CLOSE, IM_END, IM_START, OPEN, MonTok, _fields, monitor, policy, quiet,
)
from test_window_edge import FixedDrafter  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read"}}]
E1 = {"E1": {"rule": "empty_turn", "mode": "act", "close": "\n\nact now.\n</think>\n\n",
             "nudges": ["call a tool", "call a tool NOW"], "tries": 2}}
PROMPT = [2, 40, 17, 66, 30]


def _natural(prompt, think_on: bool, budget: int = 0, n: int = 30):
    """What the model writes for the chat request, raw: its tokens."""
    serve(FixedDrafter(97, 5, tree_mode=False))
    app.STATE["tok"] = MonTok()
    think = ThinkBudget(MonTok(), budget=budget)
    with quiet():
        return list(app.generate_stream(torch.tensor(prompt), n, set(), think))


def _prompt(think_on: bool) -> list[int]:
    return MonTok().apply_chat_template([{"role": "user", "content": "go"}],
                                        enable_thinking=think_on)["input_ids"][0].tolist()


def _ask(end, mode: str = "on", tasks=E1, stream=True, think_on=True, fmt="tags",
         budget=None, max_tokens=40, tok=None, spec=None, **extra):
    """One chat request whose model ends at `end` (a token, or a list of them);
    (monitor, content, reasoning, finish). `tok` and `spec`: a family's tokenizer (the monitor
    reads with it too, as it reads with the engine's) and its spec, in place of MonTok's."""
    serve(FixedDrafter(97, 5, tree_mode=False))
    # the answer reserve off (--answer-reserve 0): this tests the monitor's own acts, at a limit
    # where the reserve for a request with nothing asked (ENG-468 / M1.5) would close first
    app.STATE["answer_reserve"] = 0
    app.STATE["tok"] = tok or MonTok(ends=end if isinstance(end, list) else [end])
    app.STATE["cfg_eos"] = end
    app.STATE["reasoning_format"] = fmt
    if spec is not None:
        app.STATE["spec"] = spec
    mon = app.STATE["monitor"] = (monitor(mode, tasks) if tok is None
                                  else Monitor(mode, policy(tasks), tok, Journal(echo=False)))
    # SRV-617: the format named, as an unnamed raw client otherwise gets `reasoning_content`
    body = dict({"messages": [{"role": "user", "content": "go"}], "stream": stream,
                 "max_tokens": max_tokens, "tools": TOOLS, "reasoning_format": fmt,
                 "chat_template_kwargs": {"enable_thinking": think_on}}, **extra)
    if budget is not None:
        # the server's own --think-budget: ThinkBudget's streamed phrase. A client's budget
        # closes hidden under an acting monitor (tests/test_client_budgets.py)
        app.STATE["think_budget"] = budget
    with quiet():
        head, raw = Req("/v1/chat/completions", body).response()
    content, reasoning = _fields(raw, stream)
    if stream:
        fin = [json.loads(ln[6:])["choices"][0]["finish_reason"] for ln in raw.splitlines()
               if ln.startswith("data: {") and json.loads(ln[6:])["choices"]][-1]
    else:
        fin = json.loads(raw)["choices"][0]["finish_reason"]
    return mon, content, reasoning, fin


def _first_new(seq, lo: int) -> tuple[int, int]:
    """(position, token) of the first token from `lo` on that did not occur before it."""
    i = next(i for i in range(lo, len(seq)) if seq[i] not in seq[:i])
    return i, seq[i]


def _hidden_in_ctx(text: str) -> bool:
    ids = MonTok().encode(text)
    ctx = list(app.STATE["last_ctx"])
    return any(ctx[i:i + len(ids)] == ids for i in range(len(ctx) - len(ids) + 1))


# ------------------------------------------------------------------ 1: inside the reasoning

def test_1_the_end_token_inside_the_reasoning_block():
    out = _natural(_prompt(True), True)
    _, stop = _first_new(out, 3)                   # the model "ends" inside its reasoning
    for fmt in ("tags", "reasoning_content", "both"):
        for stream in (True, False):
            mon, content, reasoning, fin = _ask(stop, fmt=fmt, stream=stream)
            acts = mon.journal.actions()
            assert acts and acts[0]["reason"] == "empty_turn(case=A)", (fmt, stream, acts)
            assert "act now" not in content + reasoning, (fmt, stream)
            assert _hidden_in_ctx("\n\nact now.\n"), "the close phrase was forwarded"
            if fmt == "reasoning_content":
                assert "</think>" not in content + reasoning and reasoning
            else:
                assert content.count("</think>") == 1, (fmt, content)
                assert content.split("</think>")[1], "the answer continues after the close"


class Hy4MonTok(MonTok):
    """MonTok with Hy4-flash's tags in place of Qwen's, for the real `Hy4Tokenizer` to wrap: `<think:opensource>`
    at OPEN, `</think:opensource>` at `close` (its character, if any, is read as `?`), the arg-key form's tags and
    its wrapper on ids MonTok leaves free. The template ends as the checkpoint's does: `<think:opensource>` with
    nothing after it, `<think:opensource></think:opensource>` for `no_think`."""

    def __init__(self, close: int, ends=()):
        super().__init__(ends)
        free = [i for i in (86, 87, 88, 89, 94, 95, 96, CLOSE, 0) if i != close]
        self.close = close
        self.special = {TAGS["<think>"]: OPEN, TAGS["</think>"]: close, "<|im_end|>": IM_END,
                        "<|im_start|>": IM_START}
        for name in [n for n in list(TAGS.values()) + list(WRAPPERS) if n not in self.special]:
            self.special[name] = free.pop(0)

    def __len__(self):
        return 97

    def convert_tokens_to_ids(self, text):
        return self.special.get(text, -1)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            tag = next((s for s in self.special if text.startswith(s, i)), None)
            if tag is not None:
                out.append(self.special[tag])
                i += len(tag)
                continue
            k = ALPHA.find(text[i]) + 1
            out.append(k if k > 0 and k != self.close else ALPHA.index("?") + 1)
            i += 1
        return out

    def decode(self, seq, skip_special_tokens=True):
        names = {v: k for k, v in self.special.items()}
        out = []
        for t in map(int, seq):
            if skip_special_tokens and (t in self.ends or t in (IM_END, IM_START)):
                continue
            out.append(names[t] if t in names else ALPHA[t - 1] if 1 <= t <= len(ALPHA) else "#")
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, reasoning_effort="high", **kw):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        text += "<|im_start|>assistant\n" + TAGS["<think>"]
        if reasoning_effort == "no_think":
            text += TAGS["</think>"]
        return {"input_ids": torch.tensor([self.encode(text)])}


def test_1_a_close_as_the_first_token_is_an_answer_not_the_reasoning():
    """SRV-955, Hy4-flash in Chat (2026-11-23): its prompt opens the block (`<think:opensource>`), the model
    closed it as its first token and answered "ready", and the client got `ready</think>\\n\\nready`. The loop
    showed the first token to the budget only while no block was open, so the close was never seen; at the end
    token the block still read as open and the answer as empty, and E1 case A forwarded its close: the closer
    visible in the answer, and the answer again after it. Here with Hy4-flash's tags through its own tokenizer
    and its spec, the close the first token the model writes (the stand-in's id for it), then a short answer,
    then the end token: no act, the answer once, every format, streamed and not."""
    probe = Hy4MonTok(CLOSE)
    prompt = probe.apply_chat_template([{"role": "user", "content": "go"}])["input_ids"][0].tolist()
    assert prompt[-1] == OPEN, prompt                         # the prompt ends inside the block
    out = _natural(prompt, True)
    close = out[0]                                            # the model's first token is its close
    i, stop = _first_new(out, 3)                              # then an answer, then its end token
    assert all(1 <= t <= len(ALPHA) and t != close for t in out[1:i]), (close, out[:i + 1])
    tok = Hy4Tokenizer(Hy4MonTok(close, ends=[stop]))
    assert tok.apply_chat_template([{"role": "user", "content": "go"}], enable_thinking=True)[
        "input_ids"][0].tolist() == prompt, "the close's id changed the prompt"
    assert tok.convert_tokens_to_ids("</think>") == close and tok.decode([close]) == "</think>"
    answer = tok.decode(out[1:i])
    for fmt in ("tags", "reasoning_content", "both"):
        for stream in (True, False):
            mon, content, reasoning, fin = _ask(stop, fmt=fmt, stream=stream, tok=tok, spec=HY4.spec)
            assert mon.journal.actions() == [], (fmt, stream, mon.journal.actions(), content)
            assert "act now" not in content + reasoning and not _hidden_in_ctx("\n\nact now.\n"), (fmt, stream)
            assert app.STATE["last_ctx"][len(prompt):len(prompt) + i] == out[:i], (fmt, stream)
            assert fin == "stop" and reasoning == "", (fmt, stream, fin, reasoning)
            if fmt == "reasoning_content":
                assert content == answer, (fmt, stream, content, answer)
            else:
                assert content.count("</think>") == 1 and content.split("</think>")[1] == answer, (
                    fmt, stream, content, answer)


# ------------------------------------------------------------------ 2: after the reasoning

def test_2_an_empty_answer_after_the_clients_own_close():
    """The engine's own budget closes the block (streamed phrase, ThinkBudget's), and the
    model's very next token is its end token: a hidden turn, the answer goes on."""
    out = _natural(_prompt(True), True, budget=3, n=200)
    phrase = MonTok().encode(ThinkBudget.PHRASE)
    k = next(i for i in range(len(out)) if out[i:i + len(phrase)] == phrase) + len(phrase)
    stop = out[k]
    assert stop not in out[:k], "the token after the phrase must be new to end there"
    for stream in (True, False):
        mon, content, reasoning, fin = _ask(stop, stream=stream, budget=3, max_tokens=200)
        acts = mon.journal.actions()
        assert acts and acts[0]["reason"] == "empty_turn(case=B)", acts
        assert acts[0]["text"] == "call a tool"
        assert _hidden_in_ctx("<|im_end|>\n<|im_start|>user\ncall a tool<|im_end|>\n"
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert content.count("<think>") == 1, "no second <think> reaches the client"
        assert content.split("</think>")[1].strip(), "the answer goes on after the nudge"


def test_2_an_empty_answer_with_thinking_off():
    out = _natural(_prompt(False), False)
    stop = out[0]                                  # the end token as the first token
    mon, content, _, fin = _ask(stop, think_on=False)
    acts = mon.journal.actions()
    assert acts and acts[0]["reason"] == "empty_turn(case=B)", acts
    assert "<think>" not in content and content.strip(), content


# ------------------------------------------------------------------ 3: GIVE_UP

def test_3_two_failures_end_in_give_up():
    """Every token the model draws ends it: the first act nudges, the second nudges harder,
    the third finish is GIVE_UP -- one sentence, finish stop, journaled gave_up."""
    for stream in (True, False):
        mon, content, _, fin = _ask(list(range(97)), think_on=False, stream=stream)
        evs = mon.journal.events()
        acts = [e for e in evs if e["sp"] == "SP2"]
        assert [(e["decision"], e["text"]) for e in acts] == [
            ("CONTINUE_HIDDEN", "call a tool"), ("CONTINUE_HIDDEN", "call a tool NOW"),
            ("GIVE_UP", policy().texts["give_up"])], acts
        assert content == policy().texts["give_up"] and fin == "stop", (content, fin)
        assert evs[-1]["decision"] == "OUTCOME" and evs[-1]["outcome"] == "gave_up", evs[-1]


# ------------------------------------------------------------------ 4-7: nothing to do

def _answer_stop() -> int:
    """A token the model writes five tokens into its answer, thinking off."""
    out = _natural(_prompt(False), False)
    return _first_new(out, 5)[1]


def test_4_a_text_answer_is_left_alone():
    mon, content, _, fin = _ask(_answer_stop(), think_on=False)
    assert mon.journal.events() == [] and fin == "stop" and content, (content, fin)


def test_5_no_tools_no_action():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, tools=[])
    assert mon.journal.events() == [] and content == "" and fin == "stop"


def test_6_tool_choice_none_is_respected():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, tool_choice="none")
    assert mon.journal.events() == [] and content == "" and fin == "stop"


def test_7_shadow_only_records():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], mode="shadow", think_on=False)
    evs = mon.journal.events()
    assert evs[0]["mode"] == "shadow" and evs[0]["decision"] == "CONTINUE_HIDDEN", evs
    assert mon.journal.actions() == [] and content == "" and fin == "stop"


def test_n_above_one_and_stop_strings_pass():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False, stream=False, n=2)
    assert mon.journal.actions() == [], "n > 1 never acts"
    # a stop string that matches at once: the client asked for that stop
    mon, content, _, fin = _ask(_answer_stop(), think_on=False, stop=[MonTok().decode(out[:1])])
    assert mon.journal.events() == [] and fin == "stop"


def test_the_outcome_says_what_came_of_the_act():
    out = _natural(_prompt(False), False)
    mon, content, _, fin = _ask(out[0], think_on=False)
    end = mon.journal.events()[-1]
    assert end["decision"] == "OUTCOME" and end["outcome"] in ("text", "empty", "gave_up"), end
    assert end["acts"] >= 1 and end["hidden_tokens"] > 0
    assert len(end["detail"]["resume_ms"]) == end["acts"], end


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
