"""Hy4-flash's tags under the names the server's parsers read (engine/models/hy4_flash/tokenizer.py).

Without the checkpoint's files: the renaming on a stand-in tokenizer, a call in the model's own form read by
the arg-key parser whole and streamed, the effort words. With HY4_TOKENIZER (a directory with the checkpoint's
tokenizer.json, tokenizer_config.json and chat_template.jinja): the real template and ids -- thinking on and
off, the effort line, the tools prompt, an earlier turn's calls written back in the model's form.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.hy4_flash.tokenizer import TAGS, WRAPPERS, Hy4Tokenizer, effort_of, plain, real  # noqa: E402
from server.api import toolformats  # noqa: E402
from server.api.toolcall import ToolCallBuffer, parse_tool_calls  # noqa: E402
from server.stream import opens_think  # noqa: E402

DIR = os.environ.get("HY4_TOKENIZER")
SKIP = None if DIR and os.path.isfile(os.path.join(DIR, "tokenizer.json")) else "skipped: HY4_TOKENIZER names no tokenizer"
S = ":opensource"
CALL = (f"<tool_calls{S}><tool_call{S}>write_file<arg_key{S}>path</arg_key{S}><arg_value{S}>a.txt</arg_value{S}>"
        f"<arg_key{S}>content</arg_key{S}><arg_value{S}>line 1\nline <2></arg_value{S}><arg_key{S}>n</arg_key{S}>"
        f"<arg_value{S}>3</arg_value{S}></tool_call{S}><tool_call{S}>get_time</tool_call{S}></tool_calls{S}>")
TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}]
TYPES = {"write_file": {"path": frozenset({"string"}), "content": frozenset({"string"}), "n": frozenset({"integer"})}, "get_time": {}}
WANT = [("write_file", {"path": "a.txt", "content": "line 1\nline <2>", "n": 3}), ("get_time", {})]


class _Stand:
    """A tokenizer of single characters and the model's tags (ids from 1000)."""

    unk_token_id = 0

    def __init__(self):
        self.tags = list(TAGS.values()) + list(WRAPPERS)

    def convert_tokens_to_ids(self, t):
        if isinstance(t, str):
            return 1000 + self.tags.index(t) if t in self.tags else 0
        return [self.convert_tokens_to_ids(x) for x in t]

    def __call__(self, text, **kw):
        ids, i = [], 0
        while i < len(text):
            for k, tag in enumerate(self.tags):
                if text.startswith(tag, i):
                    ids.append(1000 + k)
                    i += len(tag)
                    break
            else:
                ids.append(ord(text[i]) + 2000)
                i += 1
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, **kw):
        return "".join(self.tags[i - 1000] if i < 2000 else chr(i - 2000) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return kw


def _calls(calls) -> list:
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def _read(tok, ids) -> None:
    fmt = toolformats.get("argkey")
    text = tok.decode(ids)
    assert ":opensource" not in text and "<tool_calls" not in text, text
    content, calls = parse_tool_calls(fmt.rewrite("Writing it." + text, eos=True), eos=True, types=TYPES)
    assert _calls(calls) == WANT and content == "Writing it.", (calls, content)
    for size in (1, 2, 5, len(ids)):
        b = ToolCallBuffer(names=["write_file", "get_time"], types=TYPES, fmt=fmt)
        out, sent = "", ""
        for i in range(size, len(ids) + size, size):
            now = tok.decode(ids[:i])
            out += "".join(b.feed(now[len(sent):]))
            sent = now
            b.drain_deltas()
        left, _ = b.finish(eos=True)
        assert _calls(b.calls) == WANT and not (out + left).strip(), (size, b.calls, out, left)


def test_the_tags_get_their_plain_names():
    tok = Hy4Tokenizer(_Stand())
    assert plain(real("<think>a</think><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>")) == \
        "<think>a</think><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
    assert tok.convert_tokens_to_ids("</think>") == tok._tok.convert_tokens_to_ids(f"</think{S}>") >= 1000
    assert tok("</think>").input_ids == [tok.convert_tokens_to_ids("</think>")]
    assert tok.decode(tok(f"x<think{S}>y</think{S}>z").input_ids) == "x<think>y</think>z"
    assert opens_think(tok.decode(tok(f"a<think{S}>").input_ids)) and not opens_think(tok.decode(tok(f"a<think{S}></think{S}>").input_ids))
    try:
        Hy4Tokenizer(type("T", (), {"convert_tokens_to_ids": lambda self, t: 0, "unk_token_id": 0})())
        raise SystemExit("a tokenizer without the tags was wrapped")
    except ValueError:
        pass
    return f"{len(TAGS)} tags, 2 wrappers"


def test_a_call_in_the_models_form_reads_as_arg_key():
    tok = Hy4Tokenizer(_Stand())
    _read(tok, tok._tok(CALL).input_ids)
    return "two calls in one turn, whole and streamed in pieces of 1, 2 and 5 tokens"


def test_the_effort_words():
    tok = Hy4Tokenizer(_Stand())
    assert effort_of({}) == "high" and effort_of({"enable_thinking": False, "reasoning_effort": "xhigh"}) == "no_think"
    # the server hands the template's own word (the family's EffortSpec): a mode passes, anything else is high
    assert [effort_of({"reasoning_effort": w}) for w in ("low", "medium", "high", "xhigh", "max", "no_think", "odd")] == \
        ["high", "high", "high", "high", "max", "no_think", "high"]
    kw = tok.apply_chat_template([], enable_thinking=False, tools=TOOLS)
    assert kw == {"reasoning_effort": "no_think", "tools": TOOLS}
    return "off -> no_think, the template's own modes pass, anything else -> high; enable_thinking never reaches the template"


def _real():
    from transformers import AutoTokenizer
    return Hy4Tokenizer(AutoTokenizer.from_pretrained(DIR))


def test_the_real_template_thinking_on_and_off():
    if SKIP:
        return SKIP
    tok = _real()
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]
    on = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
    off = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    xh = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh", tokenize=False)
    mx = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, reasoning_effort="max", tokenize=False)
    o, c = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    on, off = list(on["input_ids"] if hasattr(on, "keys") else on), list(off["input_ids"] if hasattr(off, "keys") else off)
    assert (o, c) == (120029, 120030) and on[-1] == o and off[-2:] == [o, c], (on[-3:], off[-3:])
    assert opens_think(tok.decode(on[-8:])) and not opens_think(tok.decode(off[-8:]))
    assert tok("</think>", add_special_tokens=False).input_ids == [c]
    raw_on, raw_off = tok._tok.decode(on), tok._tok.decode(off)
    assert "reasoning_effort:high" in raw_on and "reasoning_effort:no_think" in raw_off and "reasoning_effort:max" in mx and "reasoning_effort:high" in xh
    return f"on ends with {o}, off with {o} {c}; the effort line reads high, no_think and max (a word not its own: high)"


def _closed_stream(tok, close: str, prompt: list, reasoning: str, answer: str, fmt: str) -> tuple[list, list]:
    """A budget-closed turn as the server streams it: the prompt opens the block (the template's generation prompt),
    the model reasons, the server writes `close` (ThinkBudget's forced close, the ids the model reads next), the model
    answers. Returns (the forced ids, the (field, text) pieces the reasoning split hands out, piece by piece)."""
    from engine.spec import ThinkBudget
    from server.stream import Detokenizer, Reasoning, opens_think
    tb = ThinkBudget(tok, 4, phrase=close).start(prompt)
    assert tb.inside, "the template's generation prompt opens the block"
    r = tok(reasoning, add_special_tokens=False).input_ids
    tb.observe(r)
    forced = list(tb.close_ids)
    tb.observe(forced)
    assert tb.done and not tb.inside, "the forced close ends the block for the budget"
    a = tok(answer, add_special_tokens=False).input_ids
    split = Reasoning(fmt, in_think=opens_think(tok.decode(prompt[-8:], skip_special_tokens=False)))
    det = Detokenizer(lambda seq: tok.decode(seq, skip_special_tokens=True))
    ids, out = [], []
    for t in r + forced + a:
        ids.append(t)
        out += split.push(det.push(ids))
    out += split.push(det.flush(ids)) + split.finish()
    return forced, out


def test_a_forced_close_ends_the_reasoning_where_the_answer_starts():
    """SRV-893 class (the scouting run on 7f2a41f5: on the raw route a budget-closed Hy4-flash turn's answer held
    reasoning, once the vendor's close phrase itself): the family's spec closes a budget-ended block the way its
    template ends a finished one, `</think:opensource>` and the answer straight after (no sentence, no blank lines).
    With the pack's tokenizer: the forced ids are that one token, the reasoning field is the model's reasoning to the
    close, the next text the split sees is the answer, and in `tags` the block's text ends with the close."""
    if SKIP:
        return SKIP
    from engine.models import REGISTRY
    from engine.spec import ThinkBudget
    tok = _real()
    close = REGISTRY["hy_v4_flash"].spec.reasoning.close
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2 + 2?"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
    prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)
    reasoning, answer = "The user asks for 2 + 2. That is", "2 + 2 = 4."
    end = tok._tok.convert_tokens_to_ids(f"</think{S}>")
    forced, out = _closed_stream(tok, close, prompt, reasoning, answer, "reasoning_content")
    assert forced == [end], (forced, [tok._tok.convert_ids_to_tokens(t) for t in forced])
    got = {f: "".join(t for k, t in out if k == f) for f in ("reasoning", "content")}
    assert got == {"reasoning": reasoning, "content": answer}, got
    fields = [k for k, t in out if t]
    assert fields == sorted(fields, key=("reasoning", "content").index), f"answer text inside the reasoning: {out}"
    _, tagged = _closed_stream(tok, close, prompt, reasoning, answer, "tags")
    block = "".join(t for k, t in tagged if k == "tagged")
    assert block.endswith("</think>") and block[:-len("</think>")] == reasoning, block
    assert "".join(t for k, t in tagged if k == "content") == answer, tagged
    # the shared close (every other family's) on this tokenizer: the vendor's sentence, then the same end token, then
    # blank lines the template never writes after it; it stays every other family's close
    old = ThinkBudget(tok, 4).close_ids
    assert old[-2] == end and tok.decode(old[-1:]) == "\n\n" and "Considering the limited time" in tok.decode(old), old
    return f"the forced close is {forced} ({tok._tok.convert_ids_to_tokens(end)}); the next text is the answer"


def test_the_real_template_tools_and_a_call_round_trip():
    if SKIP:
        return SKIP
    tok = _real()
    msgs = [{"role": "user", "content": "Write it."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": {"path": "a.txt", "content": "line 1\nline <2>", "n": 3}}},
                {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "tool", "tool_call_id": "c2", "content": "12:00"}]
    text = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    assert CALL in text and '"name": "write_file"' in text and f"<tool_response{S}>ok</tool_response{S}>" in text
    # the turn the template wrote is what the model writes: its ids read back as the two calls
    ids = tok._tok(CALL, add_special_tokens=False).input_ids
    assert len(ids) < 60 and tok._tok.convert_tokens_to_ids(f"<tool_call{S}>") in ids
    _read(tok, ids)
    return f"the tools prompt, an earlier turn's calls in the model's form, and {len(ids)} ids read back as two calls"


#: the template's turn marks (its lines 3-5), by name
TURN = ("<｜hy_start:opensource｜>", "<｜hy_middle:opensource｜>", "<｜hy_end:opensource｜>")


class _TurnStand(_Stand):
    """The stand-in with the template's turn marks as tokens too."""

    def __init__(self):
        super().__init__()
        self.tags += list(TURN)

    def encode(self, text, add_special_tokens=False):
        return self(text).input_ids


def test_the_turn_marks_reach_the_servers_lookup():
    """SRV-905: `template_marks` gives the turn opener's id, the ids of `user` and the turn end's id, ids the
    wrapped tokenizer knows, and the server's lookup (`_marks_of`) returns them; a tokenizer without the marks
    gives none. With HY4_TOKENIZER: the pack's ids (120000, `user` as one token, 120025)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_forward_tree  # noqa: F401  (the CPU environment before the server imports)
    from server import app
    stand = _TurnStand()
    tok = Hy4Tokenizer(stand)
    want = (stand.convert_tokens_to_ids(TURN[0]), stand.encode("user"), stand.convert_tokens_to_ids(TURN[2]))
    assert tok.template_marks() == want and min(want[0], want[2]) >= 1000, tok.template_marks()
    assert app._marks_of(tok) == want
    assert Hy4Tokenizer(_Stand()).template_marks() is None and app._marks_of(Hy4Tokenizer(_Stand())) is None
    if SKIP:
        return SKIP
    real = _real()
    marks = real.template_marks()
    assert marks == (120000, [3717], 120025) and app._marks_of(real) == marks, marks
    ids = real.apply_chat_template([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                                   add_generation_prompt=True, enable_thinking=False)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    i = next(j for j in range(len(ids)) if ids[j] == marks[0] and ids[j + 1:j + 2] == marks[1])
    assert ids[i - 1] == marks[2], "the first user turn starts right after the system turn's end"
    return f"the pack's marks {marks}; the first user turn at row {i}"


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
