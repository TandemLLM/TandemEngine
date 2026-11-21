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
  silent_tags            the wrapper's tags by their plain names with the model's ids, which the token lookup takes
                         too: a forced tool_choice lets the model write them around its calls (ENG-472, the family's
                         `ServeSpec.call_wrapper`), and the text still leaves them out
  apply_chat_template    the checkpoint's own template, untouched, so the prompt holds the model's real tags and
                         the wrapper around an earlier turn's calls. Thinking is the template's `reasoning_effort`:
                         `no_think` when the request turns thinking off (the template then closes the block in the
                         prompt), else the mode the server hands it, one of the template's own (the family's
                         EffortSpec in arch.py maps the effort levels onto them), else `high`, its default.

Everything else is the wrapped tokenizer's.
"""

from __future__ import annotations

SUFFIX = ":opensource"
#: plain tag -> the model's token text
TAGS = {f"<{slash}{name}>": f"<{slash}{name}{SUFFIX}>" for name in ("think", "tool_call", "arg_key", "arg_value") for slash in ("", "/")}
#: the wrapper around a turn's calls: written by the model and by the template, not part of the arg-key form
WRAPPERS = (f"<tool_calls{SUFFIX}>", f"</tool_calls{SUFFIX}>")
#: ENG-472: the wrapper's plain names (arch.py's `call_wrapper`) -> the model's token text, for the token lookup only:
#: the text never holds them, so neither `plain` nor `real` writes them
WRAPPER_TAGS = dict(zip(("<tool_calls>", "</tool_calls>"), WRAPPERS))
_LOOKUP = {**TAGS, **WRAPPER_TAGS}
#: the modes the pack's template takes as `reasoning_effort` (its chat_template.jinja, lines 38-46)
MODES = ("max", "high", "no_think")


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
    """The template's `reasoning_effort` for a request's template arguments: `no_think` with thinking off, the mode
    the server handed when it is one of the template's, else `high`."""
    if kwargs.get("enable_thinking") is False:
        return "no_think"
    word = kwargs.get("reasoning_effort")
    return word if word in MODES else "high"


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
            return self._tok.convert_tokens_to_ids(_LOOKUP.get(tokens, tokens))
        return self._tok.convert_tokens_to_ids([_LOOKUP.get(t, t) for t in tokens])

    @property
    def silent_tags(self) -> dict:
        """ENG-472: the tags the text leaves out, by their plain names, with the model's ids (engine/grammar.py
        `Vocab.from_tokenizer`)."""
        return {p: self._tok.convert_tokens_to_ids(r) for p, r in WRAPPER_TAGS.items()}

    def apply_chat_template(self, messages, *args, **kw):
        kw = dict(kw)
        kw["reasoning_effort"] = effort_of(kw)
        kw.pop("enable_thinking", None)
        return self._tok.apply_chat_template(messages, *args, **kw)

    def template_marks(self):
        """SRV-905: the chat template's turn marks for the server's lookup (server/app.py `_marks_of`): the turn
        opener's id, the ids of `user` and the turn end's id, by their names in the template (its lines 3-5:
        `<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>...<｜hy_end:opensource｜>`). None when the
        wrapped tokenizer does not hold them. The ids only: the text renaming (`real`) knows none of these."""
        try:
            start = self._tok.convert_tokens_to_ids("<\uff5chy_start:opensource\uff5c>")
            end = self._tok.convert_tokens_to_ids("<\uff5chy_end:opensource\uff5c>")
            user = [int(t) for t in self._tok.encode("user", add_special_tokens=False)]
        except Exception:                                          # noqa: BLE001
            return None
        unk = getattr(self._tok, "unk_token_id", None)
        if not (isinstance(start, int) and isinstance(end, int) and start >= 0 and end >= 0 and user
                and unk not in (start, end)):
            return None
        return start, user, end
