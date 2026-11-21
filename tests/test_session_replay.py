"""tools/session_replay.py on a CPU, with no engine: the session as it was (the defaults), the agent shape ENG-474 adds
(twelve tools, several calls a scripted turn, parallel calls asked for, the model's own calls answered, the calls'
summary), and the stream read back into the JSON answer's shape (a local server with a canned stream).

Run: python tests/test_session_replay.py
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tools import session_replay as R  # noqa: E402

DEFAULTS = dict(base="http://x/v1", out="", compare=None, system_chars=2000, first_files=2, piece_chars=500, turns=4,
                turn_chars=300, after_turns=3, summary_tokens=8, summary_chars=200, chat_turns=3, chat_chars=100,
                max_tokens=32, x_tandem="", tools=None, calls_per_turn=1, parallel_tool_calls=False, stream=False,
                extra="", own_calls=False, expect_calls=False)


def _run(answer, **kw) -> tuple[dict, list]:
    sent = []

    def post(base, body):
        sent.append(json.loads(json.dumps(body)))
        return dict(answer(body), _client_ms=1.0)

    old = R._post, R._post_stream, R._get
    R._post = R._post_stream = post
    R._get = lambda url: {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rep = R.run(argparse.Namespace(**dict(DEFAULTS, **kw)))
    finally:
        R._post, R._post_stream, R._get = old
    return rep, sent


def _text(body):
    return {"choices": [{"message": {"content": "A summary."}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}


def test_the_defaults_send_the_session_as_before():
    rep, sent = _run(_text)
    assert [r["kind"] for r in rep["requests"]] == ["first"] + ["turn"] * 4 + ["compaction", "post", "after", "date",
                                                                               "after"] + ["chat"] * 3
    for b in sent:
        assert set(b) <= {"model", "messages", "tools", "max_tokens", "temperature", "stream"} and b["stream"] is False, b
        if b.get("tools"):
            assert [t["function"]["name"] for t in b["tools"]] == [t["function"]["name"] for t in R._tools()]
        for m in b["messages"]:
            if m["role"] == "assistant":
                assert [c["function"]["name"] for c in m["tool_calls"]] == ["read"], m
    assert len(R._tools()) == 9
    return f"{len(sent)} requests, nine tools, one scripted call a turn, no field added"


def _calls(body):
    """The stand-in model: two calls (the second with arguments that are no JSON object on a `bash` call)."""
    return {"choices": [{"message": {"content": "", "reasoning_content": "Reading two.", "tool_calls": [
        {"id": f"m{len(body['messages'])}a", "type": "function", "function": {"name": "read", "arguments": '{"filePath": "/a"}'}},
        {"id": f"m{len(body['messages'])}b", "type": "function", "function": {"name": "glob", "arguments": '{"pattern": "*"}'}}]},
        "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 40}}


def test_the_agent_shape():
    rep, sent = _run(_calls, tools=12, calls_per_turn=3, parallel_tool_calls=True, own_calls=True,
                     extra='{"reasoning_effort": "low"}', chat_turns=0)
    first = sent[0]
    assert len(first["tools"]) == 12 and first["tools"][-1]["function"]["name"] == "chrome-devtools_navigate_page"
    assert first["parallel_tool_calls"] is True and first["reasoning_effort"] == "low"
    scripted = [m for m in first["messages"] if m["role"] == "assistant"]
    assert len(scripted) == 2 and all(len(m["tool_calls"]) == 3 for m in scripted)
    assert sum(m["role"] == "tool" for m in first["messages"]) == 6
    # every later turn answers the model's own two calls: its assistant message, then a result for each
    nxt = sent[1]["messages"]
    own = [m for m in nxt if m["role"] == "assistant"][-1]
    ids = [f"m{len(first['messages'])}a", f"m{len(first['messages'])}b"]
    assert [c["id"] for c in own["tool_calls"]] == ids and own["reasoning_content"] == "Reading two.", own
    assert [m["tool_call_id"] for m in nxt[-2:]] == ids
    c = rep["calls"]
    assert c["requests"] == len(sent) - 1 and c["called"] == c["several"] == c["requests"] and c["max_calls"] == 2
    assert c["failures"] == 0, c
    # an answer with no call, a call left as text, arguments that are no object and no token: each counted
    bad = iter([_text, lambda b: {"choices": [{"message": {"content": "<tool_call>read</tool_call>"},
                                               "finish_reason": "stop"}], "usage": {"completion_tokens": 9}},
                lambda b: {"choices": [{"message": {"tool_calls": [{"id": "z", "function": {"name": "read", "arguments": "[1]"}}]},
                                        "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 9}},
                lambda b: {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}])
    rep, _ = _run(lambda b: next(bad, _calls)(b), turns=3, after_turns=0, chat_turns=0, own_calls=True)
    c = rep["calls"]
    assert c["no_call"] == [0, 1, 3] and c["call_text"] == [1] and c["bad_args"] == [2] and c["zero_tokens"] == [3], c
    return "12 tools, three calls a scripted turn, the model's own calls answered, each failure counted"


class _SSE(http.server.BaseHTTPRequestHandler):
    EVENTS = [{"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Two "}}]},
              {"choices": [{"index": 0, "delta": {"reasoning_content": "files."}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c0", "type": "function",
                                                                   "function": {"name": "read", "arguments": ""}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"filePath": '}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "c1", "type": "function",
                                                                   "function": {"name": "glob", "arguments": '{"pattern": "*"}'}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"/a"}'}}]}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "timings": {"prompt_n": 7}},
              {"choices": [], "usage": {"prompt_tokens": 40000, "completion_tokens": 30}}]
    seen = []

    def do_POST(self):
        self.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in self.EVENTS:
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        self.wfile.write(b": a comment\n\ndata: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_the_stream_reads_back_as_the_answer():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = R._post_stream(f"http://127.0.0.1:{srv.server_address[1]}/v1", {"model": "m", "messages": [], "stream": False})
    finally:
        srv.shutdown()
    assert _SSE.seen[0]["stream"] is True and _SSE.seen[0]["stream_options"] == {"include_usage": True}
    msg = out["choices"][0]["message"]
    assert msg["reasoning_content"] == "Two files." and msg["content"] == ""
    assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]] == \
        [("c0", "read", {"filePath": "/a"}), ("c1", "glob", {"pattern": "*"})]
    assert out["choices"][0]["finish_reason"] == "tool_calls" and out["usage"]["prompt_tokens"] == 40000
    r = R._record("turn", out)
    assert r["calls"] == ["read", "glob"] and not r["calls_bad"] and r["finish"] == "tool_calls" and r["forwarded"] == 7
    return "two calls from their deltas, the reasoning, the finish, the usage and the timings"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
