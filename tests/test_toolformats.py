"""ENG-407 (ENG-E17): every tool-call form behind one interface, the contract each must keep.

For every registered `ToolFormat` (server/api/toolformats) and every family's `ServeSpec.tool_format`:

  1  the family's form is registered; the default is the XML form
  2  a call rendered in the form parses back to the same name and arguments: whole (`rewrite` +
     `parse_tool_calls`) and streamed through the ToolCallBuffer in pieces of 1 to N characters, with
     the form's stream rewriter in front when it has one; a typed argument comes back typed
  3  the forced-call constraint accepts the form's forced rendering, and a call of another form is not
     accepted where the two differ
  4  C4's line shows the offered tool and argument and no placeholder; a broken block's reason names none
  5  the argument loop guard reads the open call's name
  6  ENG-581: the XML and arg-key forms join calls and a document (the request's or SET_CONSTRAINT's), the
     Pythonic form answers 400

CPU. Run: python tests/test_toolformats.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine import grammar as G  # noqa: E402
from engine.models import REGISTRY  # noqa: E402
from server.api import toolformats  # noqa: E402
from server.api.toolcall import ToolCallBuffer, diagnose, parse_tool_calls  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                     "n": {"type": "integer"}}, "required": ["path", "content"]}}}]
TYPES = {"write_file": {"n": frozenset({"integer"})}}
ARGS = {"path": "/tmp/a b.txt", "content": "line one\nline two", "n": 3}


def _formats():
    toolformats.get(None)
    return dict(toolformats.FORMATS)


def test_every_family_form_is_registered():
    fmts = _formats()
    assert toolformats.get(None).name == toolformats.DEFAULT and toolformats.DEFAULT in fmts
    for name, a in REGISTRY.items():
        assert a.spec.tool_format in fmts, (name, a.spec.tool_format)
    try:
        toolformats.get("no-such-form")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown form must fail")
    return f"{len(fmts)} forms: {sorted(fmts)}"


def _stream(fmt, text: str, size: int) -> list:
    b = ToolCallBuffer(names=["write_file"], types=TYPES, fmt=fmt)
    rw = fmt.stream_rewriter()
    content = ""
    for i in range(0, len(text), size):
        piece = text[i:i + size]
        content += "".join(b.feed(rw.feed(piece) if rw is not None else piece))
        b.drain_deltas()
    if rw is not None:
        content += "".join(b.feed(rw.finish(eos=True)))
    left, _ = b.finish(eos=True)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in b.calls], content + left


def test_a_rendered_call_reads_back_in_every_form():
    for name, fmt in _formats().items():
        text = "Writing it." + fmt.render("write_file", ARGS)
        content, calls = parse_tool_calls(fmt.rewrite(text, eos=True), eos=True, types=TYPES)
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
        assert got == [("write_file", ARGS)] and content == "Writing it.", (name, got, content)
        for size in (1, 2, 3, 7, len(text)):
            calls, content = _stream(fmt, text, size)
            assert calls == [("write_file", ARGS)] and content == "Writing it.", (name, size, calls, content)
    return f"{len(_formats())} forms, whole and streamed"


def test_the_forced_constraint_takes_the_forms_own_rendering():
    fmts = _formats()
    for name, fmt in fmts.items():
        pat, typed = fmt.forced_regex(TOOLS, ["write_file"], many=False, grammar=G)
        assert typed == {"write_file": {"n"}}, (name, typed)
        d = G.compile_regex(pat)
        own = fmt.forced_render("write_file", ARGS)
        assert d.accept[d.walk(0, own.encode())], (name, own)
        for other in fmts.values():
            alt = other.forced_render("write_file", ARGS)
            if alt != own:
                assert not d.accept[d.walk(0, alt.encode())], (name, other.name)
        assert set(fmt.markers) <= set(G.CALL_MARKERS) or name != toolformats.DEFAULT


def test_the_notes_show_a_real_tool_and_no_placeholder():
    for name, fmt in _formats().items():
        line = fmt.example("write_file", "path")
        assert "write_file" in line and "path" in line, (name, line)
        for t in (line, fmt.example("write_file"), diagnose("<function>\nx", name), diagnose("hello", name),
                  diagnose("write_file<arg_key>x</arg_key>oops", name)):
            assert "NAME" not in t and "KEY" not in t, (name, t)


def test_the_block_reader_base_takes_nothing():
    r = toolformats.BlockReader()
    r.reset()
    r.stream(None, 0)
    assert r.takes("anything") is False and r.parse("x", None) is None


def test_the_loop_guard_reads_the_open_calls_name():
    for name, fmt in _formats().items():
        text = fmt.forced_render("write_file", ARGS)
        opened = text.split("<tool_call>", 1)[1][:40] if "<tool_call>" in text else ""
        assert fmt.open_call_name(opened) == "write_file", (name, opened)
        assert fmt.open_call_name("") is None


# ENG-472: the forced calls and their vocabulary as every form had them before a family could name a call wrapper
# (recorded on 7f2a41f5): a family without one (`ServeSpec.call_wrapper` empty) keeps them byte for byte. ENG-581
# re-records the arg-key form's on purpose (on 7ebed84e): its fourth body, the calls-or-document union, was the 400
# and is now its own calls beside the document; the XML and Pythonic forms' stand
FORCED_BODIES = (
    {"tools": TOOLS, "tool_choice": "required"},
    {"tools": TOOLS, "tool_choice": "required", "parallel_tool_calls": True},
    {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "write_file"}}},
    {"tools": TOOLS, "tool_choice": "auto", "response_format": {"type": "json_object"}},
)
FORCED_PIN = {"pythonic": "2ab9f22930de2d8ea1480e14d0c2c69b6227b639ced102615f5b55ad5ffd246d",
              "argkey": "ec15c9eb244d3c285fa16f20c64f4c979f581ec2a244672b2c3a2bd19a592492",
              "xml": "52fcdc8c06cf05fde77f36a271af53d8296f6a82499cb7b5de0b4e10069cd8ea"}
VOCAB_PIN = "7337e27bc882782fc16940104e0ad03c7c39a495c1915a2b2f13d264c894ece3"


class _TagTok:
    """A byte-level tokenizer (ids 0-255 the bytes, as GPT-2's pieces) with every form's tags as added ids from 256,
    text in the answer: what Vocab.from_tokenizer reads of a served tokenizer."""

    TAGS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
            "<|tool_call_start|>", "<|tool_call_end|>")
    unk_token_id = None
    all_special_ids = [264]

    def __init__(self):
        from server.logprobs import _DECODER
        self.enc = {b: c for c, b in _DECODER.items()}
        self.added_tokens_decoder = {256 + k: t for k, t in enumerate(self.TAGS)}

    def __len__(self):
        return 265

    def convert_ids_to_tokens(self, ids):
        return [self.enc[i] if i < 256 else self.added_tokens_decoder.get(i, "<eos>") for i in ids]

    def convert_tokens_to_ids(self, t):
        return 256 + self.TAGS.index(t) if t in self.TAGS else None

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.added_tokens_decoder[i] if i in self.added_tokens_decoder else
                       ("" if i == 264 else bytes([i]).decode("latin-1")) for i in ids)


def _forced_digest(fmt) -> str:
    import hashlib
    from server import compat
    rows = []
    for body in FORCED_BODIES:
        try:
            a = compat.answer_constraint(dict(body, messages=[]), True, fmt=fmt)
            rows.append(None if a is None else [a.pattern, a.kind, sorted((k, sorted(v)) for k, v in (a.types or {}).items()),
                                                a.calls, a.parts, list(a.markers)])
        except compat.Refusal as exc:
            rows.append(["refused", exc.param])
    act = compat.act_constraint({"kind": "choice", "value": ["write_file"], "scope": "tool_args", "strict": False},
                                dict(FORCED_BODIES[0], messages=[]), True, 265, fmt=fmt)
    rows.append([act.answer.pattern, list(act.answer.markers), act.same_as])
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _vocab_digest() -> str:
    import hashlib
    tok = _TagTok()
    out = []
    for fmt in _formats().values():
        for markers in ((), G.CALL_MARKERS, tuple(fmt.markers)):
            v = G.Vocab.from_tokenizer(tok, 266, markers=markers)
            out.append([b.hex() if b else None for b in v.bytes])
    return hashlib.sha256(json.dumps(out).encode()).hexdigest()


def test_a_form_without_a_wrapper_keeps_its_forced_calls():
    """ENG-472: every form's forced calls (required, parallel, named; the calls-or-document union; the extension's
    tool_args act) and the vocabulary their markers build are the bytes recorded before a family could name the tags
    its model writes around a turn's calls; every family but the one that names them names none."""
    got = {name: _forced_digest(fmt) for name, fmt in _formats().items()}
    assert got == FORCED_PIN, got
    assert _vocab_digest() == VOCAB_PIN, _vocab_digest()
    named = {n for n, a in REGISTRY.items() if getattr(a.spec, "call_wrapper", ())}
    assert named <= {"hy_v4_flash"}, named
    return f"{len(got)} forms x {len(FORCED_BODIES) + 1} constraints, one vocabulary per form's markers; wrappers: {sorted(named)}"


def test_a_form_with_its_own_calls_joins_a_document():
    """ENG-581: the XML and arg-key forms join calls and a document under auto (`joins_documents`), the Pythonic
    form keeps the 400; the extension's SET_CONSTRAINT regex beside tools joins the same way on the arg-key form,
    with the family's wrapper around the calls and the form's tags and the wrapper as markers."""
    from server import compat
    from server.api.toolformats import argkey
    fmts = _formats()
    assert {n for n, f in fmts.items() if f.joins_documents} == {"xml", "argkey"}, fmts
    body = {"messages": [], "tools": TOOLS, "response_format": {"type": "json_object"}}
    try:
        compat.answer_constraint(body, True, fmt="pythonic")
        raise AssertionError("the Pythonic form cannot join a document")
    except compat.Refusal as exc:
        assert exc.param == "tools" and "not served in this model's tool-call format" in str(exc), str(exc)
    wrapper = ("<tool_calls>", "</tool_calls>")
    for wrap in ((), wrapper):
        act = compat.act_constraint({"kind": "regex", "value": "[a-z]{1,8}", "scope": None, "strict": False},
                                    {"messages": [], "tools": TOOLS}, True, 265, fmt="argkey", wrapper=wrap)
        a = act.answer
        assert a.kind == "response_format" and a.calls and a.parts is not None, a
        assert a.markers == argkey.MARKERS + wrap and act.same_as == {"structured_outputs": {"regex": "[a-z]{1,8}"}}
        d = G.compile_regex(a.pattern)
        for text, ok in (("abc", True), (fmts["argkey"].forced_render("write_file", {"path": "a", "content": "b"}), True),
                         ("<tool_calls><tool_call>write_file<arg_key>path</arg_key><arg_value>a</arg_value><arg_key>"
                          "content</arg_key><arg_value>b</arg_value></tool_call></tool_calls>", bool(wrap)),
                         ("<tool_call>\n<function=write_file>\n</function>\n</tool_call>", False), ("", False)):
            assert bool(d.accept[d.walk(d.start, text.encode())]) == ok, (wrap, text)
    return "xml and argkey join, pythonic refuses; the act's regex joins the arg-key calls, with and without the wrapper"


def test_a_schema_the_calls_cannot_read_is_a_400():
    """ENG-581's review: a function whose `parameters.properties` is not an object, or whose `required` is null, a
    number, a bool or a list holding objects, stopped every form's `forced_regex` with a TypeError (a 500) under
    `tool_choice` required, a named function and auto beside a document; now each is the 400 on `tools`. What was
    served still is: an empty list for no properties, a `required` string, list of numbers or object. Only the
    functions the calls may name are read."""
    from server import compat
    fmts = _formats()
    ok = {"type": "function", "function": {"name": "ok", "parameters": {"type": "object", "properties": {}}}}
    bad = {"props a list": {"properties": ["a"]}, "props a string": {"properties": "a"},
           "required null": {"properties": {"a": {"type": "string"}}, "required": None},
           "required a number": {"properties": {"a": {"type": "string"}}, "required": 1},
           "required a bool": {"properties": {"a": {"type": "string"}}, "required": True},
           "required objects": {"properties": {"a": {"type": "string"}}, "required": [{"a": 1}]}}
    served = [{"properties": []}, {"properties": {"a": {"type": "string"}}, "required": "a"},
              {"properties": {"a": {"type": "string"}}, "required": [1]},
              {"properties": {"a": {"type": "string"}}, "required": {"a": 1}}]
    bodies = {"required": {"tool_choice": "required"},
              "named": {"tool_choice": {"type": "function", "function": {"name": "f"}}},
              "auto + document": {"response_format": {"type": "json_object"}}}
    seen = 0
    for name, fmt in fmts.items():
        for label, params in bad.items():
            tools = [{"type": "function", "function": {"name": "f", "parameters": dict(params, type="object")}}, ok]
            for mode, extra in bodies.items():
                if mode == "auto + document" and not fmt.joins_documents:
                    continue                                   # the form's own 400 comes first
                try:
                    compat.answer_constraint(dict(extra, messages=[], tools=tools), True, fmt=fmt)
                except compat.Refusal as exc:
                    assert exc.param == "tools" and "function 'f'" in str(exc), (name, label, mode, str(exc))
                    seen += 1
                else:
                    raise AssertionError(f"{name} {label} {mode}: served")
            # a named call to the readable function does not read the other one
            a = compat.answer_constraint({"messages": [], "tools": tools, "tool_choice": {
                "type": "function", "function": {"name": "ok"}}}, True, fmt=fmt)
            assert a is not None and a.kind == "tool_choice", (name, label)
        for params in served:
            tools = [{"type": "function", "function": {"name": "f", "parameters": dict(params, type="object")}}]
            for mode, extra in bodies.items():
                if mode != "auto + document" or fmt.joins_documents:
                    got = compat.answer_constraint(dict(extra, messages=[], tools=tools), True, fmt=fmt)
                    assert got is not None, (name, params)
    return (f"{seen} 400s on tools over {len(fmts)} forms; a named call reads its own function only; "
            f"{len(served)} odd shapes served as before")


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
