# Hy4-flash

Tencent's Hy4-flash has about 220B parameters, and 64B of them are Engram n-gram tables. About 11B run for each token. The engine serves it from two of our EXL3 packs on one DGX Spark, up to 8 chats at once on lanes that take turns, text only. This page lists what the engine does with it, the limits a user meets, and what is not measured yet.

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

Up to 8 chats at once (ENG-485). Each lane is a side slot with the whole 262,144-token window on the shared KV pool: a lane's latents and indexer keys take pages only for the rows it holds, and an idle conversation goes to the park and the NVMe spill under memory pressure. The lanes take turns, a whole round each, with no joint pass yet, so two chats at once share the decode rate. The profiles run the round runner without its pipe (`QSE_RUNNER_PIPE=0`): with the pipe the drafter accepted fewer tokens a round on this family and one chat decoded 5.6 to 6.6 % slower on `hy4-3.56`, with the same text.

The window is 262,144 tokens. Both packs ran at it through the server, and a run filled `hy4-3.08` to 262,144 real tokens. Each token of state costs 16,896 bytes, 4.13 GiB at the full window. The checkpoint reads 1,048,576 tokens, but no run went past 262,144.

The prompt pass runs on fixed-order kernels by default (`QWEN38_HY4_PREFILL_V2=1`, SPD-327; `0` keeps the pass before). One request at a time on an idle box, a fresh server, the first request of each kind, no drafter, the experts on the shared prefill kernel in both arms. The pass before was measured on 2026-11-20; the fixed-order pass on 2026-11-21 with every knob at its default (prefill tok/s is the prompt's tokens over the time to the first token):

| pack | prompt | first token, pass before | first token, fixed-order | prefill tok/s, before | prefill tok/s, fixed-order |
|-|-|-|-|-|-|
| `hy4-3.56` | chat, 3,565 tokens | 13.71 s | 5.90 s | 260 | 604 |
| `hy4-3.56` | docs, 31,852 tokens | 120.57 s | 46.31 s | 264 | 688 |
| `hy4-3.08` | chat, 3,565 tokens | 13.53 s | 5.12 s | 264 | 696 |
| `hy4-3.08` | docs, 31,852 tokens | 120.60 s | 45.24 s | 264 | 704 |

Decode did not change: 22.6 to 23.1 tok/s plain on `hy4-3.56`, 23.5 to 24.3 on `hy4-3.08`. The flip moves the prompt pass's bits once, so its answers differ from the pass before's; a drafted answer still equals the plain one on both packs. On chat2048 (`hy4-3.08`, 16 rows) it scores KL 0.10151 and top-1 88.35 % against 0.10093 and 88.49 % before, a gap below the 0.003 to 0.005 between two correct forwards of this model, and the next-token rule passes on both rows it checks. Every cache tier (snapshot, resident prefix, disk tier, spill) equals a cold read with it on, on both packs (2026-11-21, on a tree from before the lanes moved the state onto the KV pool; to run again on the current head).

The prompt pass reads every row in one summation order, whatever the chunk it is in (R83). Its packed projections run on `tools/exl3_rows.py`: one order a weight, eight K slices of KT/8 tiles (S1 w8), on the skinny GEMM up to 1,024 rows (in 128-row calls past 128) and on the decoded slab's fused product above, a 3.5-bit projection (the twelve indexer query projections of `hy4-3.56`) on the slab at every row count. Before, a projection took the row kernel up to 32 rows, the skinny GEMM at the shape's own split up to 128 and the library GEMM above, each summing in its own order, so a resume that cut the prompt off the 4,096-row grid read other bits than a cold read. The row pass and the drafter keep their kernels: decode and verify are unchanged. A one-row prompt chunk runs the prompt pass, not a decode step (`Capabilities.prompt_rows`). The step moves the prompt pass's bits once, as the fixed-order flip did. Gate 0's figures, 2026-11-21, the order put on the engine by the gate's scripts before this code, to be replaced by the step's own box run: with the trunk in that order, chunks cut at 349 and 74, 390 and 33, 1,000 and 2,000, in chunks of 257, and as 2,999 rows then one gave a cold read's head inputs, last-row logits and state on both packs; the 31,852-token prompt in the engine alone took 42.06 and 42.09 s against 45.48 and 44.18 s before on `hy4-3.56` (0.952 of before) and 41.60 and 41.42 s against 44.99 and 43.40 s on `hy4-3.08` (0.954); 64 decode steps after it ran at 23.6 to 23.7 tok/s against 23.5 to 23.6 (`hy4-3.56`) and 24.5 to 24.7 against 24.4 to 24.7 (`hy4-3.08`). A 4,096-row chunk's trunk products took 756 ms against 960 ms on the library GEMM (`hy4-3.56`), and the 1,024-row line is where the slab first beat the skinny GEMM.

Before the fixed-order pass, the prompt pass ran a chunk's routed experts on the shared prefill expert kernel (`QWEN38_HY4_PREFILL_EXPERTS=group`, the default), and both passes still do from 128 rows a chunk; a shorter chunk takes the row kernels, which are faster there (the measured crossover: 128 rows on `hy4-3.56`, 256 on `hy4-3.08`). On the shared kernel a chunk's expert outputs equal the row kernels' on all 41 mixture-of-experts layers of both packs (`tools/hy4_check.py --only experts-equal`, 82 of 82 layer-chunks); its first-token times and prefill rates are the table's "pass before" columns. With `rows` they run on the family's row kernels, in their grouped form, which decodes an expert's tiles once for up to 8 of its pairs (`QWEN38_HY4_PREFILL_GROUPED=1`, the default; `0` keeps the pair kernels). The row kernels' two forms were measured before the shared kernel came in: on `hy4-3.56`, one request at a time on an idle box, the first token of a 3,565-token chat came after 19.1 to 24.2 s grouped against 27.0 to 29.5 s on the pair kernels, and that of a 31,852-token prompt after 176.0 to 187.6 s against 253.2 to 256.4 s. The answers were the same bytes in all six pairs, the drafter kept the same tokens, and decode ran at 22.5 to 23.0 tok/s plain on both. On 8,192 rows of real text the two kernels gave equal last-row logits, head inputs and state, and equal expert outputs in all 41 MoE layers. A 262,144-token prompt took 1,735 s (151 tok/s) in the bare engine, on an older row path.

A chat keeps its state between turns in the same caches as on other models. Anchors, prefix snapshots in the state store, the disk tier and the session spill give back the state a cold prefill writes, bit for bit. So does a turn that resumes at the end of the last prompt, at any row and with a one-row tail (R83: the prompt pass reads a row in one order whatever the cut). A turn that resumes at the end of the last answer or from a session snapshot passes a weaker test, as on every model: it must write the same greedy tokens (`docs/exactness.md`). Verify blocks and decode steps wrote those rows, and they compute attention, the value half, the indexer and the router with other arithmetic than the prompt pass, so those rows are not a cold read's. Here is how the caches fit this model:

- A turn that holds the chat's whole last prompt never runs as a guest, also after the chat comes back from the park or the NVMe. A guest is a short request served over a chat's rows, which go back afterwards, and it keeps nothing. When a pinned head comes back the engine does not know the last prompt's length, and the next turn follows the size rule once. With thinking on and no tools the template drops the last answer's reasoning, so the next prompt holds only the last prompt. That turn resumes at the end of the last prompt and reads the last answer and the new message. With thinking off, or with tools when the client sends the reasoning back, a turn resumes at the end of the last answer and reads only the new message.
- The conductor's own questions run as guests of the chat at any length. They keep no session snapshot, and the chat keeps its end. A question that finds nothing in place gives its rows back after it, or at the next request when its client left first.
- The engine cuts a prompt without a conductor plan where its first user turn starts, when its system turn holds at least 256 tokens. The state store keeps the state at that row, and the next prompt with the same system turn resumes there. The conductor's helper question then reads its system turn again only when the store does not hold it: after a start, and after the memory policy shed the store, which is the first cache it sheds. Two prompts whose system turns share 256 tokens or more count as one prompt's head, and the newer takes the older's place at the back of the store's eviction order. Past the mark anchors sit every 4,096 tokens. With a mark under 4,096 tokens the disk tier keeps no anchor of that prompt: it writes only the 4,096-token lines below the mark.
- A state store entry holds up to 0.25 GiB on this model, a prompt of about 11,000 tokens, unless the start sets `--store-entry-gb`.
- The end of the last prompt lives in the lane only. A chat with thinking on and no tools that loses the lane reads itself again once when it comes back, from its first user turn on.
- A resumed prefill forwards at least one row, which reads the draft head's picks again, so the first round after the engine resumes can draft. The `[drafter]` line counts the rounds that drafted nothing because the picks were stale (`stale=`, since the start). No device run has read that count yet.

On the device, on both packs, each tier's restore equals a cold read: a snapshot, the resident prefix with and without a decode step, a disk flush, the disk tier with and without a step, and the spill (2026-11-21, on the tree before the lanes). None of the turn resumes above has run on the device yet, on either pack: at the end of the last answer or of the last prompt, beside a guest or the conductor's question, and from the head mark. Nor has the `stale=` count been read.

Hy4-flash needs its chat template. It always opens with a system turn that carries `reasoning_effort:<mode>`. Without that turn the unquantized model scores much worse: perplexity 21.0 against 7.25 on technical text, 10.9 against 3.79 on code, 10.5 against 7.07 on a novel. A think block that does not match the mode hurts the same way, and raw text without the template scores far worse. The template takes three modes: `no_think`, `high` (its default) and `max`. The family's spec (`engine/models/hy4_flash/arch.py`) maps the engine's effort levels onto them, and the shared effort policy does the rest, as for every model:

| level | reached by | template mode | thinking cap |
|-|-|-|-|
| off | the client, or the server default `off` | `no_think` | none: no thinking |
| off | a conductor category (`summarize`) | `high`; the router closes the block at the end of the prompt | none: no thinking |
| low | the client, a category | `high` | 1,024 tokens |
| medium | the client, a category, or nobody on the engine alone (no conductor plan that decides the effort) | `high` | 8,192 |
| high | the client, a category | `high` | 16,384 |
| xhigh | a category | `high` | none below the conductor's backstop of 30,720 |
| xhigh or max | the client | `max` | none on the engine alone and where the client's word wins, and the block then closes 2,048 tokens and its close before the output limit; under the agent profiles (`hermes`, `coding`, `overnight`) the class's cap |

When a budget ends the block on the engine alone, the engine closes it as the template ends a finished one: `</think:opensource>`, and the answer follows at once (the family's spec names its close, `ReasoningSpec.close`). The shared close is Qwen's: the vendor's sentence, the tag and blank lines, which the template never writes. On the raw route the scouting run on build `7f2a41f5` found budget-closed Hy4 answers holding reasoning, once the sentence itself. Under the conductor the session monitor closes with the vendor's sentence hidden from the client, the same tokens in the model's context, and that answer was clean at 651 tokens; the monitor's close is not changed. Which part of the old close made the raw answers hold reasoning is not isolated. A request that names no effort, on the engine alone, thinks in mode `high` and stops at 8,192 thinking tokens, with the answer reserve after them; the server's `--think-budget` wins when it is set. `--reasoning-effort` keeps one job here: `off` turns thinking off for a request that says nothing about thinking. Any other value is not used, and the start log says so. Under the conductor, a category decides the cap and never the mode, and whether a client's word wins over the category is the profile's own rule (TandemApp's `thinking.client_word`).

Only a client's own `xhigh` or `max` reaches `max`. The lengths below were measured; the mode each turn ran in was read from the code of the build that served it, not observed. On build `c98758af`, which sent the server's default xhigh to `max`, three such turns thought 82, 7,759 and 25,806 tokens; the 25,806-token turn ran 853.8 s and came from a category's xhigh with a server default of xhigh. Four Chat turns with an explicit `low`, which that build rendered as `high`, thought 45 to 536 tokens. Tencent's later template drops `max`; a pack built from it would need its spec to map xhigh to `high`.

Tool calls use the model's own tags, which the tokenizer reads under the `argkey` form's names. One served call passed on `hy4-3.56`: the answer ended with `finish_reason` `tool_calls` and one well-formed call, and the model answered the follow-up turn that carried the tool's result. That is one call and one follow-up, so the model card says `tools: true`, and `parallel_tool_calls` and `structured` stay false: neither is tested. The model opens its calls with its own `<tool_calls>` wrapper, which the text leaves out. A forced `tool_choice` (required, named, or required with `parallel_tool_calls: true`) lets the model write that opener and its closer as its own tokens around the calls, as it does each call's tags (the family's `call_wrapper`, ENG-472). Before, the constraint masked the opener and forced `<tool_call>` first. That is checked on the CPU with the pack's tokenizer: the template's own calls walk the constraint to its end and read back as the calls. A served forced call with these tags has not run yet.

Vision is not served. Each pack holds the tower at 5 bits, and the engine reads text only.

Memory: a start with serve.env's default limits takes the pack's own: a ceiling 12 GiB over what the allocator holds after the load, rounded up to a whole GiB, and a peak 7 GiB above that (`engine/models/hy4_flash/serving.py` `memory_limits`). Since the lanes, the window's state (16,896 bytes a token, 4.13 GiB at 262,144 tokens) lives in the KV pool, whose pages are mapped outside the allocator, so the load holds less and the pack's own limits are lower. On build `70c250f6` (measured 2026-11-21) they were 81 and 88 GiB on `hy4-3.56` and 72 and 79 GiB on `hy4-3.08`. Before the lanes, `hy4-3.56` held 72.5 to 72.7 GiB allocated after the load and took 85 and 92. The memory policy works with those limits plus what the side slots added when they were built. A configured ceiling or peak (serve.env's, a profile's) also loses slot 0's window, which the pool holds outside the allocator; the pack's own limits were measured without it and keep it. The start's `[limits]` line names both: `MEM_CEILING_GIB_SET` and `MEM_PEAK_GIB_SET` are the pack's own limits, and `MEM_CEILING_GIB` and `MEM_PEAK_GIB` the policy's. With the side slots' 3.09 and 3.18 GiB of `70c250f6` that is 84.09 and 91.09 GiB on `hy4-3.56` and 75.18 and 82.18 on `hy4-3.08` (computed, not yet read from a start). Build `70c250f6` took the window off the own limits too and ran with 79.97 and 86.97 and with 71.05 and 78.05, about 8 GiB of room for the caches in place of 12. The profiles ask for 97 GiB (`hy4-3.08`) and 101 GiB (`hy4-3.56`) of free memory before a start. Those two figures are the measured load plus the server's 7.3 GiB prefill estimate plus the 15 GiB floor, not a measured start.

## Speed

Both packs ran in one session on 2026-11-20, on an idle box with the full 262,144-token window and each pack's own memory limits. Every figure is one prose chat request through the server: a 3,565-token prompt in the pack's template with thinking off, 600 tokens, greedy, tok/s from token 50, two requests a server, one sequence. This is not the measured table's shape, and the model card's table is empty until that run exists.

| profile | plain, tok/s | DFly2 + lookup + StairCut, tok/s | tokens a drafted round | first token | load, start to healthy |
|-|-|-|-|-|-|
| `hy4-3.56` | 22.63, 23.00 | 27.81, 28.37 | 2.26 | 27.0 to 31.5 s | 95 to 105 s, 147 to 165 s |
| `hy4-3.08` | 23.78, 24.12 | 32.08, 33.02 | 2.53 | 23.6 to 29.6 s | 87 to 93 s, 138 to 144 s |

In every run the drafted text equals the plain text. The two packs write different answers to the prompt, and why a round of `hy4-3.08` keeps more tokens on it was not isolated. These runs read the prompt on the pair kernels, the default when they were measured; the prompt path has changed since (above), and decode does not depend on it.

## License

The weights are Tencent's, under the Tencent Hy License, Version 1.0, which each pack carries as `LICENSE`. It is derived from Apache 2.0, and it counts quantized versions as derivative works, so the packs carry its terms. Its Section 10 adds terms for large Model-as-a-Service businesses: above US$50 million in revenue or 100 million monthly active users, a separate license from Tencent is needed. These terms apply to every recipient of the packs. TandemEngine itself is AGPL-3.0 or commercial.

## What is not measured

- The measured table's runs (`tools/parallel_card.py`, raw `/v1/completions`) on either pack.
- A budget-closed raw turn on the end-tag close: whether the answer then holds no reasoning, on either pack (the CPU tests show the ids the model reads and the split, not what the model writes next).
- `hy4-3.08` at the full window on an idle box, on the current build.
- Thinking on: speed and how many drafted tokens a round keeps.
- Structured output, parallel tool calls, and tool calls beyond the one call and its follow-up.
- Quality past 2,048 tokens, task-level checks, and any comparison with Tencent's own stack or its FP8 release.
- The free memory a start needs: the profiles' figures come from arithmetic.
- Whether the table reader's rate caps the prompt pass. It reads a chunk's Engram rows on one thread before the layers, in one-submit slices of 256 (SPD-319). On both packs its rows equal the reads it replaced (`tools/hy4_check.py --only table-rows`). Under the fixed-order pass a 4,096-row chunk's reads take 465 to 535 ms. Inside a loaded engine the reader managed 20,000 to 37,000 reads a second on `hy4-3.56` and 41,000 to 100,000 on `hy4-3.08`, against 106,000 to 120,000 idle. Whether that limits the prefill rate was not tested.
- The fixed-order pass's quality on sets other than chat2048. Its chunk independence on the device was measured by Gate 0 (R83, above), with the trunk pinned and in the one prompt order.
- The R83 step's own run on the device: the kernel guard (`tests/gpu/test_linear_rowinv_gpu.py`), the cuts, chat2048 and the served speed. The figures above are Gate 0's.
