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

One chat at a time. This family builds no side slots yet, so the Setup Guide caps the lanes at 1 and a second request waits in the queue. While the request ahead is inside its reasoning block, the waiting line and the live view say "another request is thinking". The state takes no KV pool pages, so the profiles run with the pool off.

The window is 262,144 tokens. Both packs ran at it through the server, and a run filled `hy4-3.08` to 262,144 real tokens. Each token of state costs 16,896 bytes, 4.13 GiB at the full window. The checkpoint reads 1,048,576 tokens, but no run went past 262,144.

Reading a prompt is slow. The prompt pass runs a chunk's routed experts on the grouped kernels, which decode an expert's tiles once for up to 8 of its pairs (`QWEN38_HY4_PREFILL_GROUPED=1`, the default; `0` keeps the pair kernels). On `hy4-3.56`, one request at a time on an idle box, the first token of a 3,565-token chat came after 19.1 to 24.2 s with them against 27.0 to 29.5 s on the pair kernels, and that of a 31,852-token prompt after 176.0 to 187.6 s against 253.2 to 256.4 s. The answers were the same bytes in all six pairs, the drafter kept the same tokens, and decode ran at 22.5 to 23.0 tok/s plain on both. On 8,192 rows of real text the two kernels gave equal last-row logits, head inputs and state, and equal expert outputs in all 41 MoE layers. A 262,144-token prompt took 1,735 s (151 tok/s) in the bare engine, on an older row path.

A chat keeps its state between turns in the same caches as on other models. Anchors, prefix snapshots in the state store, the disk tier and the session spill give back the state a cold prefill writes, bit for bit. A turn that resumes at the end of the last answer, at the end of the last prompt or from a session snapshot passes a weaker test, as on every model: it must write the same greedy tokens (`docs/exactness.md`). Here is how the caches fit this model:

- A turn that holds the chat's whole last prompt never runs as a guest, also after the chat comes back from the park or the NVMe. A guest is a short request served over a chat's rows, which go back afterwards, and it keeps nothing. When a pinned head comes back the engine does not know the last prompt's length, and the next turn follows the size rule once. With thinking on and no tools the template drops the last answer's reasoning, so the next prompt holds only the last prompt. That turn resumes at the end of the last prompt and reads the last answer and the new message. With thinking off, or with tools when the client sends the reasoning back, a turn resumes at the end of the last answer and reads only the new message.
- The conductor's own questions run as guests of the chat at any length. They keep no session snapshot, and the chat keeps its end. A question that finds nothing in place gives its rows back after it, or at the next request when its client left first.
- The engine cuts a prompt without a conductor plan where its first user turn starts, when its system turn holds at least 256 tokens. The state store keeps the state at that row, and the next prompt with the same system turn resumes there. The conductor's helper question then reads its system turn again only when the store does not hold it: after a start, and after the memory policy shed the store, which is the first cache it sheds. Two prompts whose system turns share 256 tokens or more count as one prompt's head, and the newer takes the older's place at the back of the store's eviction order. Past the mark anchors sit every 4,096 tokens. With a mark under 4,096 tokens the disk tier keeps no anchor of that prompt: it writes only the 4,096-token lines below the mark.
- A state store entry holds up to 0.25 GiB on this model, a prompt of about 11,000 tokens, unless the start sets `--store-entry-gb`.
- The end of the last prompt lives in the lane only. A chat with thinking on and no tools that loses the lane reads itself again once when it comes back, from its first user turn on.
- Until this family builds side slots, two chats share the one lane. A chat with an entry in the state store takes the lane when the park or the NVMe can take the chat in place: the park from 4,096 tokens above what the two chats share, the NVMe from the spill's minimum, 16,384 tokens by default. Otherwise the new chat runs as a guest, as before. A chat whose prompt matches its own session snapshot resumes from the store in any case; the chat in place then goes to the NVMe only from the spill's minimum, and below it the engine drops it, as on every model. This rule goes away with the side slots.
- A resumed prefill forwards at least one row, which reads the draft head's picks again, so the first round after the engine resumes can draft. The `[drafter]` line counts the rounds that drafted nothing because the picks were stale (`stale=`, since the start). No device run has read that count yet.

None of these resumes has run on the device yet, on either pack.

Hy4-flash needs its chat template. It always opens with a system turn that carries `reasoning_effort:<mode>`. Without that turn the unquantized model scores much worse: perplexity 21.0 against 7.25 on technical text, 10.9 against 3.79 on code, 10.5 against 7.07 on a novel. A think block that does not match the mode hurts the same way, and raw text without the template scores far worse. The template takes three modes: `no_think`, `high` (its default) and `max`. The family's spec (`engine/models/hy4_flash/arch.py`) maps the engine's effort levels onto them, and the shared effort policy does the rest, as for every model:

| level | reached by | template mode | thinking cap |
|-|-|-|-|
| off | the client, or the server default `off` | `no_think` | none: no thinking |
| off | a conductor category (`summarize`) | `high`; the router closes the block at the end of the prompt | none: no thinking |
| low | the client, a category | `high` | 1,024 tokens |
| medium | the client, a category, or nobody on the engine alone (no conductor plan that decides the effort) | `high` | 8,192 |
| high | the client, a category | `high` | 16,384 |
| xhigh | a category | `high` | none below the conductor's backstop of 30,720 |
| xhigh or max | the client | `max` | none on the engine alone and where the client's word wins, and the block then closes 2,048 tokens and the closing phrase before the output limit; under the agent profiles (`hermes`, `coding`, `overnight`) the class's cap |

A request that names no effort, on the engine alone, thinks in mode `high` and stops at 8,192 thinking tokens, with the answer reserve after them; the server's `--think-budget` wins when it is set. `--reasoning-effort` keeps one job here: `off` turns thinking off for a request that says nothing about thinking. Any other value is not used, and the start log says so. Under the conductor, a category decides the cap and never the mode, and whether a client's word wins over the category is the profile's own rule (TandemApp's `thinking.client_word`).

Only a client's own `xhigh` or `max` reaches `max`. The lengths below were measured; the mode each turn ran in was read from the code of the build that served it, not observed. On build `c98758af`, which sent the server's default xhigh to `max`, three such turns thought 82, 7,759 and 25,806 tokens; the 25,806-token turn ran 853.8 s and came from a category's xhigh with a server default of xhigh. Four Chat turns with an explicit `low`, which that build rendered as `high`, thought 45 to 536 tokens. Tencent's later template drops `max`; a pack built from it would need its spec to map xhigh to `high`.

Tool calls use the model's own tags, which the tokenizer reads under the `argkey` form's names. One served call passed on `hy4-3.56`: the answer ended with `finish_reason` `tool_calls` and one well-formed call, and the model answered the follow-up turn that carried the tool's result. That is one call and one follow-up, so the model card says `tools: true`, and `parallel_tool_calls` and `structured` stay false: neither is tested. A forced `tool_choice` (required or named) is not checked with these tags.

Vision is not served. Each pack holds the tower at 5 bits, and the engine reads text only.

Memory: after the load at the full window `hy4-3.56` holds 72.5 to 72.7 GiB allocated and 73.4 to 73.8 GiB reserved. A start with serve.env's default limits takes the pack's own: a ceiling 12 GiB over what the load holds and a peak 7 GiB above that, 85 and 92 GiB on `hy4-3.56`. The profiles ask for 97 GiB (`hy4-3.08`) and 101 GiB (`hy4-3.56`) of free memory before a start. Those two figures are the measured load plus the server's 7.3 GiB prefill estimate plus the 15 GiB floor, not a measured start.

## Speed

Both packs ran in one session on 2026-11-20, on an idle box with the full 262,144-token window and each pack's own memory limits. Every figure is one prose chat request through the server: a 3,565-token prompt in the pack's template with thinking off, 600 tokens, greedy, tok/s from token 50, two requests a server, one sequence. This is not the measured table's shape, and the model card's table is empty until that run exists.

| profile | plain, tok/s | DFly2 + lookup + StairCut, tok/s | tokens a drafted round | first token | load, start to healthy |
|-|-|-|-|-|-|
| `hy4-3.56` | 22.63, 23.00 | 27.81, 28.37 | 2.26 | 27.0 to 31.5 s | 95 to 105 s, 147 to 165 s |
| `hy4-3.08` | 23.78, 24.12 | 32.08, 33.02 | 2.53 | 23.6 to 29.6 s | 87 to 93 s, 138 to 144 s |

In every run the drafted text equals the plain text. The two packs write different answers to the prompt, and why a round of `hy4-3.08` keeps more tokens on it was not isolated. These runs read the prompt on the pair kernels; the grouped kernels, the default since, shorten the first token (above) and leave decode as it was.

## License

The weights are Tencent's, under the Tencent Hy License, Version 1.0, which each pack carries as `LICENSE`. It is derived from Apache 2.0, and it counts quantized versions as derivative works, so the packs carry its terms. Its Section 10 adds terms for large Model-as-a-Service businesses: above US$50 million in revenue or 100 million monthly active users, a separate license from Tencent is needed. These terms apply to every recipient of the packs. TandemEngine itself is AGPL-3.0 or commercial.

## What is not measured

- The measured table's runs (`tools/parallel_card.py`, raw `/v1/completions`) on either pack.
- `hy4-3.08` at the full window on an idle box, on the current build.
- Thinking on: speed and how many drafted tokens a round keeps.
- Structured output, parallel tool calls, and tool calls beyond the one call and its follow-up.
- Quality past 2,048 tokens, task-level checks, and any comparison with Tencent's own stack or its FP8 release.
- The free memory a start needs: the profiles' figures come from arithmetic.
