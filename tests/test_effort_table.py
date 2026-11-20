"""The effort table (SRV-899): what every family's request becomes for every effort a request can carry, pinned
against tests/fixtures/effort_table.json, which was written on the tree before the effort policy moved into the
families' specs. A family's prompt and budget may change only where its own spec says so.

Each row is one request through the real handler, with the fake engine in this process and the family's own
ServeSpec in force, and pins:

  1  the call build_prompt makes to apply_chat_template: the messages and every argument (a recording
     tokenizer), and the prompt ids the loop is handed                        test_the_table_without_a_plan
  2  the budget the loop is handed, the output limit, and the [think] lines   (the same rows)
  3  Hy4-flash's real template, when HY4_TOKENIZER names the pack's folder:
     the mode line the template writes and the prompt's ids                   test_the_real_hy4_template

The axes: every family of the registry; a request that carries nothing, thinking off, each effort word from
none to max at the top level and in chat_template_kwargs, a number, `reasoning.enabled` and `thinking.type`; the
server default unset, low, medium, xhigh and off; an output limit of 32,768 and of 2,048. The rows under a
conductor's plan are the pair test `test_the_table_under_a_plan` in tests/test_client_budgets.py.

Run:  python tests/test_effort_table.py
      python tests/test_effort_table.py --write FILE [--made-on ENGINE_COMMIT] [--tandemapp COMMIT] [--with-plan]
      python tests/test_effort_table.py --families qwen3_5,lfm2_moe     (check those families' rows only)
      python tests/test_effort_table.py --fixture FILE                   (check against another fixture)
Each check prints one line a family: the rows it compared and the pinned rows it skipped.
`--write` writes this tree's rows; `--with-plan` adds the plan rows (it needs TandemApp at TANDEMAPP_SRC).
"""

from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine.models import REGISTRY  # noqa: E402
from server import app, fake_engine  # noqa: E402
from server.api import thinking  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "effort_table.json")
WORDS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
REQUESTS = ([("nothing", {}), ("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}})]
            + [(f"reasoning_effort={w}", {"reasoning_effort": w}) for w in WORDS]
            + [(f"kwargs.reasoning_effort={w}", {"chat_template_kwargs": {"reasoning_effort": w}}) for w in WORDS]
            # a value no template word is: a lenient read (/v1/tokenize, the router's count) lets it through
            + [("kwargs.reasoning_effort=list", {"chat_template_kwargs": {"reasoning_effort": ["low"]}}),
               ("kwargs.reasoning_effort=dict", {"chat_template_kwargs": {"reasoning_effort": {"level": "low"}}})]
            + [("max_reasoning_tokens=512", {"max_reasoning_tokens": 512}),
               ("reasoning.enabled", {"reasoning": {"enabled": True}}),
               ("thinking.type=enabled", {"thinking": {"type": "enabled"}})])
DEFAULTS = (None, "low", "medium", "xhigh", "off")
LIMITS = (32768, 2048)
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi."}]
#: a request id in a log line
CID = re.compile(r"\b(?:chat)?cmpl-[0-9a-f]{24}\b")
#: the arguments of a call that say how, not what (the tensor form of the answer)
FORM = ("return_tensors", "return_dict", "tokenize")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Recorder:
    """The served tokenizer, with every apply_chat_template call kept: its messages and arguments."""

    def __init__(self, tok):
        self.__dict__["_tok"], self.__dict__["calls"] = tok, []

    def __getattr__(self, name):
        return getattr(self.__dict__["_tok"], name)

    def __setattr__(self, name, value):
        setattr(self.__dict__["_tok"], name, value)

    def __call__(self, *a, **k):
        return self.__dict__["_tok"](*a, **k)

    def __len__(self):
        # the grammar's vocabulary takes len() of the served tokenizer (engine/grammar.py `from_tokenizer`)
        return len(self.__dict__["_tok"])

    def apply_chat_template(self, messages, *a, **kw):
        self.__dict__["calls"].append({"messages": copy.deepcopy(messages),
                                       "kwargs": {k: v for k, v in kw.items() if k not in FORM}})
        return self.__dict__["_tok"].apply_chat_template(messages, *a, **kw)


def key(family: str, request: str, default, limit, plan: str | None = None) -> str:
    parts = [family, request, f"default={default or '-'}", f"limit={limit}"]
    return "|".join(parts + ([f"plan={plan}"] if plan else []))


#: the server's process memos a row reads through: the template's turn marks (set once from the loaded tokenizer),
#: the head's token counts and a text's token count (both remembered by bytes, not by tokenizer)
MEMOS = ("_CONV_MARKS", "_HEADS", "_TEXT_TOKENS")


@contextlib.contextmanager
def fresh():
    """The block starts with the server's process memos empty, as a fresh server of one family does, and puts them
    back after it: a row pins what that server does, not what the family served before it left behind (a head
    counted on one family's renders saves the next family's two renders of it)."""
    saved = {m: getattr(app, m) for m in MEMOS}
    app._CONV_MARKS = None
    app._HEADS = collections.OrderedDict()
    app._TEXT_TOKENS = collections.OrderedDict()
    try:
        yield
    finally:
        for m, v in saved.items():
            setattr(app, m, v)


@contextlib.contextmanager
def _served(spec, default):
    """The fake engine with `spec` in force and `default` as --reasoning-effort, for the block; yields the loop's
    calls. The server's state and its loop are put back as they were after it, and its process memos start empty
    (`fresh`)."""
    saved, real = dict(app.STATE), app.generate_stream
    try:
        with fresh():
            yield _start(spec, default)
    finally:
        app.generate_stream = real
        app.STATE.clear()
        app.STATE.update(saved)


def _start(spec, default) -> list:
    """The fake engine with `spec` in force and `default` as --reasoning-effort; returns the loop's calls. Call it
    through `_served`, which puts the server's state and loop back."""
    argv = ["--fake-engine", "--fake-tps", "200000", "--max-len", "65536"]
    if default:
        argv += ["--reasoning-effort", default]
    a = app.parser().parse_args(argv)
    app._thinking_default(a)
    app.STATE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        fake_engine.load(app, a)
    app.STATE.update(monitor=None, thinking_default=a.thinking_default, spec=spec)
    app.STATE["tok"] = Recorder(app.STATE["tok"])
    seen, real = [], app.generate_stream

    def spy(prompt, max_new, eos, think=None, *args, **kw):
        seen.append({"budget": getattr(think, "budget", None), "max_new": int(max_new),
                     "prompt": [int(t) for t in prompt.tolist()], "sup": kw.get("sup") is not None})
        yield from real(prompt, max_new, eos, think, *args, **kw)

    app.generate_stream = spy
    return seen


class _Req(app.Handler):
    """The handler without a socket."""

    def __init__(self, body, path: str = "/v1/chat/completions"):
        raw = json.dumps(body).encode()
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}
        self.path, self.command, self.request_version = path, "POST", "HTTP/1.1"
        self.requestline, self.client_address = "POST " + path, ("127.0.0.1", 0)
        self.close_connection = True


def row_of(calls: list, seen: list, out: str, head: str) -> dict:
    """What a request pins: the template calls, the prompt, the budget, the [think] lines, the status."""
    first = (calls or [{}])[0]
    kw = {k: v for k, v in (first.get("kwargs") or {}).items() if k != "tools"}
    loop = (seen or [{}])[0]
    prompt = loop.get("prompt") or []
    return {"status": head.split(" ", 2)[1] if head.startswith("HTTP/") else head[:40],
            "calls": len(calls), "calls_sha": _sha(calls), "kwargs": kw,
            "prompt_len": len(prompt), "prompt_sha": _sha(prompt),
            "budget": loop.get("budget"), "max_new": loop.get("max_new"), "monitor": loop.get("sup"),
            "think": [CID.sub("<cid>", ln) for ln in out.splitlines() if ln.startswith("[think]")]}


def rows(families=None) -> dict:
    """This tree's rows without a plan, for every family (or `families`)."""
    out = {}
    for name in sorted(families or REGISTRY):
        spec = REGISTRY[name].spec
        for default in DEFAULTS:
            with _served(spec, default) as seen:
                tok = app.STATE["tok"]
                for label, extra in REQUESTS:
                    for limit in LIMITS:
                        body = dict(copy.deepcopy(extra), messages=copy.deepcopy(MESSAGES), max_tokens=limit,
                                    stream=False)
                        tok.calls.clear()
                        del seen[:]
                        buf = io.StringIO()
                        req = _Req(body)
                        with contextlib.redirect_stdout(buf):
                            req.do_POST()
                        head = req.wfile.getvalue().partition(b"\r\n\r\n")[0].decode()
                        out[key(name, label, default, limit)] = row_of(tok.calls, seen, buf.getvalue(), head)
    return out


def real_hy4_rows() -> dict | None:
    """Hy4-flash's real template (HY4_TOKENIZER: the pack's folder): the mode line build_prompt's call writes,
    whether the prompt ends inside the reasoning block, and the ids, for every request and server default."""
    path = os.environ.get("HY4_TOKENIZER")
    if not path or not os.path.isfile(os.path.join(path, "tokenizer.json")):
        return None
    from transformers import AutoTokenizer

    from engine.models.hy4_flash.tokenizer import Hy4Tokenizer
    real = Hy4Tokenizer(AutoTokenizer.from_pretrained(path))
    out = {}
    for default in DEFAULTS:
        with _served(REGISTRY["hy_v4_flash"].spec, default):
            app.STATE["tok"] = real
            for label, extra in REQUESTS:
                body = dict(copy.deepcopy(extra), messages=copy.deepcopy(MESSAGES))
                if app.STATE.get("thinking_default") == "off":
                    body = thinking.server_default_off(body)
                # the lenient read, as the router's count and /v1/tokenize read (a bad value is left unread)
                client = thinking.read(body, app._effort_caps(), strict=False)
                ids = app.build_prompt(body, client, count="ids")
                text = real._tok.decode(ids, skip_special_tokens=False)
                modes = re.findall(r"reasoning_effort:([a-z_]+)", text)
                out[key("hy_v4_flash", label, default, "-")] = {
                    "modes": modes, "ids_sha": _sha([int(t) for t in ids]), "len": len(ids),
                    "ends": text[-40:]}
    return out


def plan_rows(families=None) -> dict:
    """The rows under a conductor's plan: tests/test_client_budgets.py builds them (a pair test)."""
    import test_client_budgets as cb
    return cb.table_under_a_plan(families)


def _load() -> dict:
    with open(FIXTURE) as fh:
        return json.load(fh)


def _check(want: dict, got: dict, families) -> tuple[list[str], dict]:
    """Every pinned row of `families` against this tree's row, and every row this tree serves for them that is not
    pinned. Returns (what differs, {family: [compared, skipped]}); a skipped row is a pinned row of a family not
    checked."""
    bad, counts = [], {}
    for k in sorted(want):
        fam = k.split("|", 1)[0]
        c = counts.setdefault(fam, [0, 0])
        if fam not in families:
            c[1] += 1
            continue
        c[0] += 1
        if k not in got:
            bad.append(f"{k}: pinned, not served on this tree")
        elif got[k] != want[k]:
            fields = sorted(f for f in set(want[k]) | set(got[k]) if want[k].get(f) != got[k].get(f))
            bad.append(f"{k}: " + "; ".join(f"{f} {want[k].get(f)!r} -> {got[k].get(f)!r}" for f in fields))
    for k in sorted(got):
        if k.split("|", 1)[0] in families and k not in want:
            bad.append(f"{k}: served on this tree, not pinned")
    return bad, counts


def _say(kind: str, counts: dict) -> None:
    """One line a family: the rows a check compared and the pinned rows it skipped."""
    for fam in sorted(counts):
        print(f"  [{kind}] {fam}: {counts[fam][0]} rows compared, {counts[fam][1]} skipped", flush=True)


#: --families: the families a check reads (the others' rows are not compared)
ONLY: set | None = None


def test_the_table_without_a_plan():
    want = _load()["rows"]
    pinned = {k.split("|", 1)[0] for k in want}
    only = ONLY if ONLY is not None else set(REGISTRY)
    missing = sorted(only - pinned)
    assert not missing, f"no pinned rows for {missing}: write them (--write) on the tree they record"
    got = rows(sorted(only))
    bad, counts = _check(want, got, only)
    _say("table", counts)
    assert not bad, f"{len(bad)} rows moved:\n" + "\n".join(bad[:40])
    return f"{sum(1 for k in want if k.split('|', 1)[0] in only)} rows, {len(only)} families"


def test_a_list_or_dict_effort_in_kwargs_tokenizes():
    """/v1/tokenize reads a request leniently: a `chat_template_kwargs.reasoning_effort` that is a list or a dict is
    no template word, so the template gets none and the count answers 200, as on main (a membership test on a dict
    raised TypeError for both, a 400). Every family's spec on this tree."""
    seen = []
    for name in sorted(REGISTRY):
        with _served(REGISTRY[name].spec, "xhigh"):
            for value in (["low"], {"level": "low"}):
                body = {"messages": copy.deepcopy(MESSAGES), "chat_template_kwargs": {"reasoning_effort": value}}
                req = _Req(body, "/v1/tokenize")
                with contextlib.redirect_stdout(io.StringIO()):
                    req.do_POST()
                head, _, raw = req.wfile.getvalue().partition(b"\r\n\r\n")
                assert head.startswith(b"HTTP/1.1 200"), (name, value, head[:60], raw[:200])
                assert json.loads(raw)["tokens"] > 0, (name, value, raw[:200])
                seen.append(name)
    return f"{len(set(seen))} families, a list and a dict each: 200"


def test_the_real_hy4_template():
    pinned = _load().get("real") or {}
    got = real_hy4_rows()
    if got is None:
        try:
            from scriptmode import Skip
        except ImportError:                                        # pragma: no cover
            Skip = RuntimeError
        raise Skip("skipped: HY4_TOKENIZER names no pack folder")
    if ONLY is not None and "hy_v4_flash" not in ONLY:
        return "not checked (--families)"
    assert pinned, "no pinned real-template rows: write them (--write with HY4_TOKENIZER set)"
    bad, counts = _check(pinned, got, {"hy_v4_flash"})
    _say("real", counts)
    assert not bad, f"{len(bad)} rows moved:\n" + "\n".join(bad[:40])
    return f"{len(got)} rows on the pack's template"


def _write(path: str, made_on: str, tandemapp: str, with_plan: bool, families=None) -> None:
    doc = {"about": "tests/test_effort_table.py: every family's template call, prompt and budget for each effort "
                    "a request can carry. Written on the tree named here; a family's rows change only with its spec.",
           "made_on": {"engine": made_on, "tandemapp": tandemapp},
           "rows": rows(families)}
    real = real_hy4_rows()
    if real is not None:
        doc["real"] = real
    if with_plan:
        doc["plan_rows"] = plan_rows(families)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}: {len(doc['rows'])} rows, {len(doc.get('plan_rows') or {})} plan rows, "
          f"{len(doc.get('real') or {})} real-template rows")


if __name__ == "__main__":
    argv = sys.argv[1:]

    def opt(name, default=None):
        if name in argv:
            i = argv.index(name)
            val = argv[i + 1]
            del argv[i:i + 2]
            return val
        return default

    FIXTURE = opt("--fixture", FIXTURE)
    fams = opt("--families")
    fams = set(fams.split(",")) if fams else None
    target = opt("--write")
    if target:
        _write(target, opt("--made-on", "unknown"), opt("--tandemapp", "-"), "--with-plan" in argv,
               sorted(fams) if fams else None)
        sys.exit(0)
    ONLY = fams
    sys.argv = sys.argv[:1]
    from scriptmode import run
    run(globals())
