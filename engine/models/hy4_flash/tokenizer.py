"""Hy4-flash's tokenizer as the server reads one: the model's own tags under the names the shared code knows.

Hy4-flash writes its reasoning and its tool calls with tags that carry a suffix, each one token:

    <think:opensource> ... </think:opensource>
    <tool_calls:opensource>
      <tool_call:opensource>NAME<arg_key:opensource>KEY</arg_key:opensource><arg_value:opensource>VALUE</arg_value:opensource>...</tool_call:opensource>
    </tool_calls:opensource>

Without the suffix this is the arg-key form the server already parses (`ServeSpec.tool_format` "argkey") and
the reasoning block it already splits. `Hy4Tokenizer` wraps the checkpoint's tokenizer and does the renaming at
the one place text and ids meet:

  decode                 the suffixed tags come out plain; the `<tool_calls>` wrapper around a turn's calls comes
                         out as nothing (each call stands alone in the arg-key form)
  encode, token lookup   a plain tag goes in as the model's token (`</think>` is the id of `</think:opensource>`)
  apply_chat_template    the checkpoint's own template, untouched, so the prompt holds the model's real tags and
                         the wrapper around an earlier turn's calls. Thinking is the template's `reasoning_effort`:
                         `no_think` when the request turns thinking off (the template then closes the block in the
                         prompt), `max` for the highest effort a client can ask for, else `high`, the template's
                         default. The model card documents `high` and `no_think`; the template also takes `max`.

Everything else is the wrapped tokenizer's.
"""

from __future__ import annotations

SUFFIX = ":opensource"
#: plain tag -> the model's token text
TAGS = {f"<{slash}{name}>": f"<{slash}{name}{SUFFIX}>" for name in ("think", "tool_call", "arg_key", "arg_value") for slash in ("", "/")}
#: the wrapper around a turn's calls: written by the model and by the template, not part of the arg-key form
WRAPPERS = (f"<tool_calls{SUFFIX}>", f"</tool_calls{SUFFIX}>")
#: a request's effort word -> the template's
EFFORT = {"xhigh": "max", "max": "max", "high": "high", "medium": "high", "low": "high", "no_think": "no_think"}


def plain(text: str) -> str:
    """The model's text with its tags under their plain names."""
    if SUFFIX not in text:
        return text
    for w in WRAPPERS:
        text = text.replace(w, "")
    for p, real in TAGS.items():
        text = text.replace(real, p)
    return text


def real(text: str) -> str:
    """Text with plain tags as the model's tokens."""
    if "<" not in text:
        return text
    for p, r in TAGS.items():
        text = text.replace(p, r)
    return text


def effort_of(kwargs: dict) -> str:
    """The template's `reasoning_effort` for a request's template arguments (`enable_thinking`, an effort word)."""
    if kwargs.get("enable_thinking") is False:
        return "no_think"
    return EFFORT.get(str(kwargs.get("reasoning_effort") or "high"), "high")


class Hy4Tokenizer:
    def __init__(self, tok):
        self._tok = tok
        missing = [r for r in list(TAGS.values()) + list(WRAPPERS) if not isinstance(tok.convert_tokens_to_ids(r), int)
                   or tok.convert_tokens_to_ids(r) in (None, getattr(tok, "unk_token_id", None))]
        if missing:
            raise ValueError(f"this tokenizer does not hold Hy4-flash's tags as tokens: {missing}")

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __len__(self) -> int:
        return len(self._tok)

    def __call__(self, text, *args, **kw):
        if isinstance(text, str):
            text = real(text)
        elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
            text = [real(t) for t in text]
        return self._tok(text, *args, **kw)

    def encode(self, text, *args, **kw):
        return self._tok.encode(real(text) if isinstance(text, str) else text, *args, **kw)

    def decode(self, ids, *args, **kw) -> str:
        return plain(self._tok.decode(ids, *args, **kw))

    def batch_decode(self, seqs, *args, **kw) -> list:
        return [plain(t) for t in self._tok.batch_decode(seqs, *args, **kw)]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._tok.convert_tokens_to_ids(TAGS.get(tokens, tokens))
        return self._tok.convert_tokens_to_ids([TAGS.get(t, t) for t in tokens])

    def apply_chat_template(self, messages, *args, **kw):
        kw = dict(kw)
        kw["reasoning_effort"] = effort_of(kw)
        kw.pop("enable_thinking", None)
        return self._tok.apply_chat_template(messages, *args, **kw)
