# Hy4-flash

Tencent's Hy4-flash has about 220B parameters, and 64B of them are Engram n-gram tables. About 11B run for each token. The engine serves it from two of our EXL3 packs on one DGX Spark, one chat at a time, text only. This page lists what the engine does with it, the limits a user meets, and what is not measured yet.

## The model

| part | Hy4-flash |
|-|-|
| layers | 42 with latent attention: 30 on a sliding window, 12 sparse (an indexer picks the key groups a token reads) |
| MLP | a dense first layer, then 512 experts a layer, 16 routed and 1 shared |
| residual stream | hyper-connection streams |
| Engram | n-gram tables at four layers, 38.11 GiB in the packs, read from the SSD a row at a time |
| state a token | 16,896 bytes (the 12 sparse layers' latents and indexer keys); the window layers keep rings of 17.7 MB a sequence |
| context | 1,048,576 tokens in the checkpoint; the engine serves 262,144 |
| draft layers | five DFly2 layers in the checkpoint |
| images | a vision tower in the pack, not served |

*Hy4-flash as the engine runs it (`engine/models/hy4_flash`).*

Its reference forward (`ref.py`) follows transformers operation for operation and matches it on tiny random models. The row path takes a block's rows together on the family's own kernels, and the load-time self-check proves on the device that a block's rows equal the decode steps bit for bit.

## The packs

| profile | repository | download | weights at load | chat2048 KL / top-1 | conv2048 KL / top-1 |
|-|-|-|-|-|-|
| `hy4-3.56` (default) | `0xBakeer/TandemLLM-Hy4-flash-EXL3-3.5` | 112.38 GB | 67.17 GiB | 0.0653 / 90.54 % | 0.0729 / 93.75 % |
| `hy4-3.08` | `0xBakeer/TandemLLM-Hy4-flash-EXL3-3.0` | 103.1 GB | 58.52 GiB | 0.0989 / 88.52 % | 0.1113 / 92.11 % |

Both packs keep attention, the shared experts, the head and the Engram tables at 5 bits. Routed experts are at 3 bits in `hy4-3.08` and 3.5 bits in `hy4-3.56`, which was also calibrated on templated chat. Both packs carry the same four table files, 40.92 GB of each download. With both packs under `~/models`, `ops/fetch_weights.py` links the tables from the first pack instead of fetching them twice.

Both repositories are private for now, so a download needs a Hugging Face token with access. The Setup Guide and `ops/use-profile.sh hy4-3.56` read the weights from `~/models/hy4-flash-exl3-3.56` (and `-3.08`), and skip the download when the folder holds every file.

Each KL column compares a pack with the unquantized BF16 checkpoint (revision `d88c9dfd`). Both sides ran through exllamav3's cache-less forward with our converter patch for this model. The reference is our forward of the BF16 weights, not Tencent's own stack, and the logits are stored in fp16. chat2048 is 16 rows of 2,048 tokens of wikitext-2 inside a templated turn with thinking off. conv2048 is 16 held-out conversations of 2,048 tokens from ultrachat. Two correct forwards of the same weights already differ by about KL 0.003 to 0.005 on such rows. On `hy4-3.08` this engine's own prompt pass scores KL 0.1009 and top-1 88.49 % on the chat2048 rows, so the engine adds nothing to the quantization there.

## Limits

One chat at a time. This family builds no side slots yet, so the Setup Guide caps the lanes at 1 and a second request waits in the queue. The state takes no KV pool pages, so the profiles run with the pool off.

The window is 262,144 tokens. Both packs ran at it through the server, and a run filled `hy4-3.08` to 262,144 real tokens. Each token of state costs 16,896 bytes, 4.13 GiB at the full window. The checkpoint reads 1,048,576 tokens, but no run went past 262,144.

Reading a prompt is slow: about 140 tok/s through the server, so the first token of a 3,565-token prompt came after 23.6 to 31.5 s. A 262,144-token prompt took 1,735 s (151 tok/s) in the bare engine, on an older row path.

The caches restore exactly, but on a chat they save less than on other models. The state store, the disk tier and the session spill each give back the state a cold prefill writes, bit for bit. Four limits come with them:

- With thinking on, the next turn of a chat reads the whole history again. The saved state holds the thinking tokens, and the next prompt leaves them out, so the two do not match. With thinking off, turn 2 of a short chat resumed 127 of its 150 prompt tokens.
- The resident prefix keeps nothing between turns on this model. The server logs `end anchor not kept (guest)`.
- Anchors sit every 4,096 tokens, so the engine reads a repeated prompt shorter than that again in full.
- After a restore the first round drafts nothing, and drafting resumes on the next round.

Hy4-flash needs its chat template. It always opens with a system turn that carries `reasoning_effort:<mode>`. Without that turn the unquantized model scores much worse: perplexity 21.0 against 7.25 on technical text, 10.9 against 3.79 on code, 10.5 against 7.07 on a novel. A think block that does not match the mode hurts the same way, and raw text without the template scores far worse. For now the engine uses one thinking mode, the template's `high`, and maps every request onto it or onto `no_think`. This effort handling is being moved onto the engine's shared effort policy, so the table below describes this build only:

| request | template mode |
|-|-|
| `enable_thinking: false`, or effort `none` or `off` | `no_think` |
| any effort from `low` to `max`, or none given | `high` |

The effort word still sets the server's reasoning cap: 1,024 tokens at `low`, 8,192 at `medium`, 16,384 at `high`, none at `xhigh` and `max`. At `high` a small question took 82 thinking tokens and the model stopped by itself. The template in these packs also takes a `max` mode, but that mode has no budget of its own: one chat turn in it thought for 7,759 tokens over 4 minutes. Tencent's later template drops `max`, and this build never asks for it.

Tool calls use the model's own tags, which the tokenizer reads under the `argkey` form's names. One served call passed on `hy4-3.56`: the answer ended with `finish_reason` `tool_calls` and one well-formed call, and the model answered the follow-up turn that carried the tool's result. That is one call and one follow-up, so the model card says `tools: true`, and `parallel_tool_calls` and `structured` stay false: neither is tested. A forced `tool_choice` (required or named) is not checked with these tags.

Vision is not served. Each pack holds the tower at 5 bits, and the engine reads text only.

Memory: after the load at the full window `hy4-3.56` holds 72.5 to 72.7 GiB allocated and 73.4 to 73.8 GiB reserved. A start with serve.env's default limits takes the pack's own: a ceiling 12 GiB over what the load holds and a peak 7 GiB above that, 85 and 92 GiB on `hy4-3.56`. The profiles ask for 97 GiB (`hy4-3.08`) and 101 GiB (`hy4-3.56`) of free memory before a start. Those two figures are the measured load plus the server's 7.3 GiB prefill estimate plus the 15 GiB floor, not a measured start.

## Speed

Both packs ran in one session on 2026-11-20, on an idle box with the full 262,144-token window and each pack's own memory limits. Every figure is one prose chat request through the server: a 3,565-token prompt in the pack's template with thinking off, 600 tokens, greedy, tok/s from token 50, two requests a server, one sequence. This is not the measured table's shape, and the model card's table is empty until that run exists.

| profile | plain, tok/s | DFly2 + lookup + StairCut, tok/s | tokens a drafted round | first token | load, start to healthy |
|-|-|-|-|-|-|
| `hy4-3.56` | 22.63, 23.00 | 27.81, 28.37 | 2.26 | 27.0 to 31.5 s | 95 to 105 s, 147 to 165 s |
| `hy4-3.08` | 23.78, 24.12 | 32.08, 33.02 | 2.53 | 23.6 to 29.6 s | 87 to 93 s, 138 to 144 s |

In every run the drafted text equals the plain text. The two packs write different answers to the prompt, and why a round of `hy4-3.08` keeps more tokens on it was not isolated.

## License

The weights are Tencent's, under the Tencent Hy License, Version 1.0, which each pack carries as `LICENSE`. It is derived from Apache 2.0, and it counts quantized versions as derivative works, so the packs carry its terms. Its Section 10 adds terms for large Model-as-a-Service businesses: above US$50 million in revenue or 100 million monthly active users, a separate license from Tencent is needed. These terms apply to every recipient of the packs. TandemEngine itself is AGPL-3.0 or commercial.

## What is not measured

- The measured table's runs (`tools/parallel_card.py`, raw `/v1/completions`) on either pack.
- `hy4-3.08` at the full window on an idle box, on the current build.
- Thinking on: speed and how many drafted tokens a round keeps.
- Structured output, parallel tool calls, and tool calls beyond the one call and its follow-up.
- Quality past 2,048 tokens, task-level checks, and any comparison with Tencent's own stack or its FP8 release.
- The free memory a start needs: the profiles' figures come from arithmetic.
