# Adding a model

TandemLLM serves models of the Qwen3.8 family: the 27B, a hybrid of Gated DeltaNet and attention layers, and Flash-Next, a mixture of experts on the same hybrid, from turboderp's EXL3 packs. LiquidAI's LFM2.5-24B-A2B (short convolutions, attention and a mixture of 64 experts, [lfm2.5-24b-a2b.md](lfm2.5-24b-a2b.md)), ships in 0.9.0 and is the first from another family. Flash-Next was the first model added through the adapter path below, and the path is written from what it took. This page lists the seams where a new model plugs in, what each one expects, and the tests a new model must pass before its numbers mean anything. Some of the engine is still specific to this family, and the last section says which parts.

## The adapter path in ten steps

A new architecture is a package under `engine/models/<name>/` with its own registry entry, and data files. Nothing under `server/`, `engine/cache.py`, `engine/disk_cache.py`, `engine/spill.py`, `engine/methods/` or `engine/settings.py` changes for it, and `tests/test_adapter_guards.py` holds every package to that: work on one model never moves another model's fingerprint. [architecture.md](architecture.md#methods-written-once-models-plugged-in) shows the layers; Flash-Next's package (`engine/models/flashnext/`) is the worked example.

1. Config. A reader that turns the checkpoint's `config.json` into the model's config object and stops on a field it does not know, naming it. Flash-Next's (`config.py`) keeps three lists: fields it reads, fields that only matter for training (`IGNORED`), and fields whose other values would change the arithmetic (`FIXED`).
2. Weights. A reader for the checkpoint's files and a `Linear` object for every weight the model multiplies by ([Weight formats](adding-a-model.md#weight-formats-the-linear-interface)). A new format needs a new class and a row-invariant kernel ([kernels.md](kernels.md)). Flash-Next's reader (`pack.py`) audits the pack before anything loads: a missing tensor, a truncated shard or a wrong sha256 stops it with the name.
3. A reference forward in torch, checked against the published implementation layer by layer. Flash-Next's (`ref.py`) equals `transformers`' `qwen4_exp` in fp32 within 5e-6 on a tiny random model, and its engine equals the official decoder layer on the served pack within BF16 noise.
4. The engine with the hot verbs: `forward`, `forward_tree`, `forward_block`, `commit_tree`, `rollback_to`, `picks`, `hidden_post_norm`, `kv.length`, `state` (the live state the cold verbs copy) and `reset`. Every verified row must equal the decode step's row bit for bit, or speculation is not lossless on it.
5. The cold verbs: `caps`, `cost()`, `cache_state()` with its sections and views, `resume_at(n)`, `settle()`, `discard_pending()`, `trim_pending(n)`, `manifest()`, `memory_parts()`, `resident_files()` and `precapture()`; with joint verify also `tree_batchable(parents)` (which blocks share a joint pass), `forward_tree_multi(items)` (the joint pass itself: every lane's block over the weights once, each row its lone verify's bits) and, optionally, `joint_policy()` (the pass's row cap, cut and rows by lane count for the weights the family serves: what its kernels keep row-invariant; without it 32 rows and the share cut) and `joint_graphs_report()` (the joint pass's captured graphs for `/health`, when the family captures them), and `joint_attn(layer, payloads)` (the graphed lanes' attention of a layer: the shared default runs a launch a lane; a family may run every lane in one). The round runner also asks a family's engine for five optional hooks, each absent by default: `prefetch_rows(items)` (a static method: the round's drafted trees as `(engine, tokens, parents)`, handed over before the cuts so host reads such as Flash-Next's PLE rows can start), `host_blocks` (a class attribute: True when the joint pass reads a block on the host, so `QSE_RUNNER_HOST_BLOCKS` hands it host tensors), `joint_capture_ok` (an attribute the runner sets before each joint pass: True when a joint graph may be captured in it, on the runner's thread with no request waiting; the family reads it and sets it back to False) `joint_stats()` (the joint pass's measured shape, for `/health` `parallel.joint_stats` and the runner's line; None when nothing is counted) and `host_timers(timed)` (with `QSE_RUNNER_COUNTERS`: the family wraps its own host calls with the runner's timer, `timed(obj, attr, name)`). A family whose joint pass leaves the host idle while the GPU runs calls `engine.methods.busy.busy()` between the pass's issue and its wait: the runner's held work (`QSE_RUNNER_BUSY_WORK`, `QSE_RUNNER_GC`) runs there; a family that never calls it keeps the runner's work where it was. The server's prefill burst runs a family's own joint prefill when its engine has `prefill_joint_ok(rows, start)` and a static `prefill_joint(items)` (each prompt's logits, streams, KV and state its own prefill's bits; optional `prefill_joint_prepare(rows, start)` runs a member's own steps on its thread first, so a failure there fails that member alone). They are not part of `ServeSpec` yet. Flash-Next keeps them in `serving.py` (`FlashNextServing`); the 27B's are installed on its class by `engine/models/qwen3_5/adapter.py`.
6. A draft head, if the checkpoint ships one. The shared MTP drafter (`engine/methods/mtp_drafter.py`) asks the head for `depth`, `reset()`, `prime()`, `capture_all()`, `launch()`, `read()` (the top-2 tokens and their probabilities a depth), `views()` and `resume(n)`; the engine offers it through `draft_heads()`. Flash-Next's head is `mtp.py`.
7. A server builder, `serve(a, ctx) -> dict`: load the weights, build the engine, the tokenizer and the drafter, and return them with the record `/health` shows.
8. The registry entry, in the package's own `arch.py` ([What a family declares](adding-a-model.md#what-a-family-declares)): `register(Arch(name, architectures, model_types, config, engine, package, serve, manifest, ...))`. `engine/models/__init__.py` finds it; nothing else is edited.
9. Data files: a StairCut price file for each profile and device (`ops/prices/<model>/<profile>.<device>.json`; without one the cut learns its prices online), the weight profiles (`ops/profiles/<name>.env`) and the Setup Guide's entry (`models/<id>/model.yaml`, [below](adding-a-model.md#register-the-model-for-the-setup-guide)).
10. Pass the tests under [The tests a new model must pass](adding-a-model.md#the-tests-a-new-model-must-pass).

```mermaid
flowchart TD
    m["methods, written once<br/>StairCut and its prices, the MTP drafter, the walk, every cache tier, the server, the governor"]
    a["model adapters, one per architecture<br/>qwen3_5: the 27B · qwen4_exp: Flash-Next · hot and cold verbs, ModelState"]
    f["weight formats<br/>the Linear protocol today; plugins with read and bind planned"]
    h["hardware, planned<br/>KernelImpl keyed by op, format and device; device profiles"]
    t["topology, planned<br/>Mesh at world 1: no collective on the code path"]
    m --> a --> f --> h --> t
    classDef planned stroke-dasharray: 5 5
    class h,t planned
```

*The engine's layers. A layer reads only the layers under it, and per-model numbers are data the adapter returns or files under `models/` and `ops/`. Filled: built; outlined: planned.*

## The seams

```mermaid
flowchart LR
    CFG[config.json] --> TC[TextConfig, engine/config.py]
    CK[checkpoint files] --> LAY[Layout, engine/loader.py]
    LAY --> W[Weights]
    W --> LIN[Linear interface, engine/linear.py]
    TC --> MODEL[target model, engine/model.py]
    LIN --> MODEL
    MODEL --> DR[drafters declare what they read]
    ENV[environment] --> SET[EngineSettings, engine/settings.py]
    SET --> MODEL
```

### Config

`engine/config.py::load_config` reads `config.json` into a `TextConfig` with every size the engine needs, from the layer count to the rotary settings. It also answers which layers are recurrent (`is_linear`, `linear_layers`, `attention_layers`). It accepts a vision-language config (the text model under `text_config`) and a text-only config (the keys at the top level).

A new family either fits `TextConfig` or extends it. The rest of the engine asks `TextConfig`, never the raw JSON.

### Layout

`engine/loader.py::Layout` says how a checkpoint names its tensors, in one place. The engine addresses every tensor by a canonical name (`embed_tokens.weight`, `lm_head.weight`, `norm.weight`, `layers.N.<module>.<proj>`), which is the language model's own naming with the checkpoint's wrapper prefix taken off.

| attribute | what it says |
|-|-|
| `prefixes` | wrapper prefixes to strip, first match wins (`model.language_model.`, then `model.`) |
| `skip_inside`, `skip_start` | tensors the language model never reads, such as a vision tower (`engine/vision.py` reads that one itself) |
| `mtp_file` | where the checkpoint keeps its prediction head |
| `embed`, `head`, `final_norm` | canonical names of the three model-level tensors |
| `projections` | the projection names a checkpoint may store as FP8 pairs or as plain matrices |

A checkpoint without `lm_head.weight` reads the embedding as its head only when the config says the two are tied. `tests/test_layout.py` builds tiny checkpoints of both kinds on the CPU and checks that they read to the same canonical names. It also pins the served checkpoint's key set: 1,250 language-model keys read, 333 vision keys skipped.

### Weight formats: the Linear interface

Every weight the engine multiplies by implements one small interface (`engine/linear.py`):

    class Linear(Protocol):
        shape: (N, K)
        nbytes: int                 # bytes read per use
        def matmul(x[..., K]) -> [..., N]

| class | format | file |
|-|-|-|
| `FP8Block` | e4m3 codes with one BF16 scale per 128 by 128 block, as the checkpoint stores them | `tools/fp8_linear.py` |
| `NVFP4Block` | e2m1 codes, an e4m3 scale per 16 weights, one fp32 scale | `tools/nvfp4_linear.py` |
| `FP8Head` | e4m3 with one fp32 scale per row, fp32 logits | `tools/head_gemv.py` |
| `BF16Block` | a plain BF16 matrix | `engine/linear.py` |

The model calls `w.matmul(x)` and never asks which format it holds. A new format is a new class with those three members, plus a kernel that keeps the row-invariance rule ([kernels.md](kernels.md)). A BF16 checkpoint loads through the same path: its projections become `BF16Block`s. `tests/test_bf16_load.py` checks that a tiny BF16 checkpoint on disk gives the same forward, bit for bit, as plain tensors.

### Settings

`engine/settings.py` is the registry of every `QWEN38_*` switch the engine reads: its default and the module that reads it. `SETTINGS.get(name)` is the only way engine code reads one, and `tests/test_settings.py` fails on any other read. `EngineSettings(env, prefix)` gives the same reads over another mapping or another prefix, which is how a profile file can be described or a second configuration built in a test.

### What a drafter reads

A drafter reads parts of the target: hidden states at some layers, the embedding, the head. Each drafter declares these through `Drafter.requires()`:

| key | meaning |
|-|-|
| `hidden_size` | the width of the hidden states its taps read |
| `tap_layers` | the target layers it reads |
| `tensors` | target tensors it reads by name, such as the head |
| `vocab_size` | the id space of its proposals |

At load, `engine.drafters.check_target` compares these with the target and refuses a mismatch by name. A DFlash2 drafter trained for another model fails there instead of drafting garbage. The lookup drafter's stores record the tokenizer's sha256 (`engine/tokfp.py`). Under another tokenizer a read skips the store with one log line, and a write refuses it.

A new model needs its own block drafter. DFlash2 drafters are trained against one target's hidden states, and a drafter from another model accepts nothing. The lookup drafter works for any model once its corpus is rebuilt with the new tokenizer (`tools/build_corpus.py`).

### The tokenizer and the chat template

The server renders prompts with the model's own chat template and reads the model's own tool-call and reasoning formats. For Qwen3.8 that means the `<think>` block, which the template opens in the prompt, and tool calls in the model's XML form (`server/api/toolcall.py`). A model with another format needs its own parser behind the same OpenAI output. `tests/fixtures/qwen38_chat_template.jinja` is the fixture the server tests render with.

### What a family declares

A family's `arch.py` registers its entry; `engine/models/__init__.py` imports every package's `arch.py` and keeps the entries in their `rank` order. Besides the import paths, an entry carries what the shared code needs to know, as data:

| field | what it says | read by |
|-|-|-|
| `spec` | `ServeSpec(tool_format, reasoning)`: the form the model writes tool calls in (a name in `server/api/toolformats/`) and `ReasoningSpec(opens_itself, effort)`: a model that writes its own `<think>`, and `EffortSpec(words, default)`, the word the template takes for each effort level and a default level of the family's own (leave it out when the template takes `low`, `medium` and `xhigh`) | the request loop, the conductor's checks, the forced-call constraint |
| `shares`, `uses` | the package's files other families build on, and the other packages' files this family builds on | the guards: a manifest or an import across packages must be declared, and a change to a shared file moves exactly its dependents' fingerprints |
| `hooks` | `EngineHooks(pdl, side_stream)`: launch options the family turns on in the shared kernels, which take them as plain arguments | the family's engine |
| `default` | the family a server without a checkpoint directory serves (one family is) | the loader |

A family's own knobs go in its package's `knobs.py` (a `KNOBS` table of the same shape as `engine/settings.py`'s), which joins the shared table at import and belongs in the family's manifest. A family's own C++ or CUDA kernel is a source file in its package: a module `engine/models/<family>/<name>.py` that compiles it keeps the sources in `engine/models/<family>/csrc/<name>/` and builds through `engine/native_build.py` ([kernels.md](kernels.md#where-the-source-lives)); the folder counts in the fingerprint with the module. A tool-call form the server does not know yet is a module in `server/api/toolformats/` behind the `ToolFormat` interface (rewrite to the JSON tier, a stream reader, the diagnosis of a broken block, the reminder's example line, the forced-call constraint); `tests/test_toolformats.py` runs the same contract on every form. TandemApp's conductor reads calls with its own copy (`tandemapp/toolformats/`), so a new form lands there too, with its cases in `contracts/engine-api-v1/toolcalls/`, which both copies must pass. A model that is the base of community packs says so in its `model.yaml` (`community_base`), and its package writes their weight profiles and checks their files (`community.py`, named by `community_base.hook`).

### The model layer: architecture registry and per-model state

`engine/models/__init__.py` says which architecture a checkpoint is. Each entry names the checkpoint's `architectures` strings and `model_type`s, a config reader and an engine class by import path, so registering a model imports nothing. `architecture(raw_config)` returns the entry, or fails with a message that names the checkpoint's architecture and every registered one.

| name | selected by | config | engine | server builder |
|-|-|-|-|-|
| `qwen3_5` | `Qwen3_5ForConditionalGeneration`, `qwen3_5_text` | `engine.config.load_config` | `engine.models.qwen3_5.adapter.Qwen38Engine` (engine/model.py's class with the adapter's cold verbs) | `engine.models.qwen3_5.serving.serve` (the default family) |
| `qwen4_exp` | `Qwen4ExpForConditionalGeneration`, `qwen4_exp_text` | `engine.models.flashnext.config.load_flashnext_config` | `engine.models.flashnext.engine.FlashNextEngine` | `engine.models.flashnext.serving.serve` |
| `lfm2_moe` | `Lfm2MoeForCausalLM`, `lfm2_moe` | `engine.models.lfm2_moe.config.load_lfm2_moe_config` | `engine.models.lfm2_moe.engine.Lfm2MoeEngine` | `engine.models.lfm2_moe.serving.serve` |
| `nemotron_h` | `NemotronHForCausalLM`, `NemotronHPuzzleForCausalLM`, the Omni architectures, `nemotron_h`, `nemotron_h_puzzle` | `engine.models.nemotron_h.config.load_nemotron_h_config` | `engine.models.nemotron_h.engine.NemotronHEngine` | `engine.models.nemotron_h.serving.serve` |
| `hy_v4_flash` | `HYV4FlashForConditionalGeneration`, `HYV4FlashForCausalLM`, `hy_v4_flash`, `hy_v4_flash_text` | `engine.models.hy4_flash.config.load_hy4_flash_config` | `engine.models.hy4_flash.engine.Hy4FlashEngine` | `engine.models.hy4_flash.serving.serve` |

The server has one loader (`server/app.py` `_load`, SRV-341). An entry's `serve` names the function that builds the model for it, `serve(a, ctx) -> dict`: the engine in the server loop's interface, the tokenizer, the drafter, the snapshot path, the device, the tree mode and accept rule, and its own record for `/health` (`weights_info`); `ctx.keys(path, weights)` sets the tokenizer's and the weights' fingerprints, `ctx.mark(stage)` writes the load ledger, `ctx.log(line)` the log. The engine says what the server may ask of it beyond the loop's verbs in `caps` (`engine/models/adapter.py` `Capabilities`: the caches and the KV pool, side slots, the cache grid, the shadow reader, joint verify); the loader turns off, with a log line, what a model cannot hold yet. Its cold verbs are `cost()`, `manifest()`, `precapture(drafter)`, `settle()`, `discard_pending()`, `trim_pending(n)`; the walk of a verified block is `engine/methods/walk.py` for every model. A model registered this way starts and answers without an edit under `server/` (`tests/test_load_shared.py` registers a third one on CPU).

`engine/models/state.py` is the cache state interface (`ModelState`), and every cache tier reads a sequence through it (`eng.cache_state()`, ENG-157). Its sections are the recurrent state by name, copied whole; its views are the position-indexed buffers with their token axis, sliced by length. It has `nbytes`, `clone`, `restore` and a signature that refuses a restore across shapes. The 27B's state is `Qwen38State`: sections `gdn.S` and `gdn.conv` (the same tensors, so a restore lands in the buffers its graphs captured), views K and V. Flash-Next's is `FlashNextCacheState` (`serving.py`): 36 GDN states and conv tails, the n-gram convolution and its token context, and `mtp.h_last` as sections (110.3 MiB); K, V and the indexer keys of its 12 attention layers as views, with the MTP layer's own rows from its drafter (29,952 bytes a token). A model without recurrent state has no sections. `tests/test_cache_any_model.py` runs a made-up state through every tier with no edit in the cache code.

A model's own code lives in its package under `engine/models/`. The disk tier's fingerprint v2 (`engine/disk_cache.py`) leaves `engine/models/` and `engine/methods/` out, together with knobs owned under `engine/models/` and `tools/fn_`, so adding a model sweeps none of the 27B's saved sessions (`tests/test_fn_fingerprint.py`). Fingerprint v3 (`engine/methods/manifest.py`) hashes what the adapter's `manifest()` lists: the model's files that compute stored rows, the knob registry, its own tile tables; the `tools/` kernels they import, every listed module's C++ and CUDA source files (`csrc/`) and the tables their knobs name follow. A new model writes its `manifest()` (Flash-Next: `engine/models/flashnext/serving.py`), and its fingerprint then moves with its kernels and tables and with nothing else (`tests/test_fp_v3.py`).

## The family contract: lanes, caches, the conductor and the effort policy

Four features every served model gets from shared code once its adapter answers a few hooks (ENG-487). Hy4-flash (`engine/models/hy4_flash/`) is the latest family to take all four this way, and it is the worked example below. This is what exists today; an architecture refactor will reshape some of it.

### Lanes and side slots

`--parallel N` serves slot 0 and N-1 side slots (server/app.py `_build_side_slots`). A side slot is a second serving object over the family's one engine, with the same weights and its own state, graphs and draft head. The family writes two verbs on its serving class: `side_slot(max_len, drafter0)`, which returns the slot's serving and drafter, and `side_slot_bytes(max_len, paged)`, which returns its bytes a token and its constant bytes, priced before anything is allocated. It also sets `Capabilities.side_slots` and `ServeSpec.side_slots`, which `tests/test_modelspec.py` holds equal. The shared pieces are in `engine/methods/side_slot.py`: `lane_drafter` (the slot's drafter over slot 0's prices and corpus), `ensure_rows` (the KV pool's pages for the rows a write reaches, never inside a capture), `side_slot_bytes` and `open_pool`. With `--kv-pool on` a lane's position-indexed views are paged (engine/kvpool.py), so every lane has the whole window and holds pages only for its rows. Memory pressure is carried by the park and the session spill, not by a smaller window.

The rest is shared. That covers the slot choice and admission (server/slots.py `Pool`, `Slot`, `Admission`, `Turn`), a resident prefix for each side slot (`_side_resident`), a short request as a guest, its promotion to the slot's conversation (`_guest_promotes`), a returning conversation's home slot (`_side_home`, `_continues_view`) and the carry into slot 0 (`_carry_from_side`). How the lanes share a round is a capability. With `joint_verify` they verify in one joint pass (`forward_tree_multi`, `tree_batchable`, `joint_policy`). Without it they take turns, a whole round each, as Hy4-flash does. `side_lane` and `joint_plain` say the rest. A family that captures graphs reports them through `graph_count()`, `graph_pools()` and `drop_graphs()`.

Tests: `tests/test_lane_contract.py` (the shared pieces against each family's own copy; a side slot shares the engine and owns its state; a family that declares a joint pass has one) and the family's own lane test (`tests/test_hy4_lanes.py`).

### The cache tiers

Every tier reads a sequence through the family's state (`cache_state()`, `engine/models/state.py` `ModelState`). Its sections are the fixed state, copied whole: recurrent states, rings, a window's rows. Its views are the position-indexed buffers, sliced by length, with their row axis. `resume_at(n)` rebuilds what derives from the committed rows. `cost()` (`CostModel`) prices a snapshot, the prefill's workspace and the family's read rate (`prefill_tok_s`, which the router's guard and the tiers use until a rate is measured, server/app.py `_read_rate_prior`). `Capabilities.anchor_grid` sets the caches' grid to the family's prefill chunk. The tiers, all in shared code:

| tier | what it keeps | where |
|-|-|-|
| resident prefix | the rows in place and their anchors on the grid; a short request runs as a guest over them and they come back | engine/cache.py `ResidentPrefix` (`is_guest`, `promote_guest`) |
| end anchor | the state at the end of the last answer, so the next turn reads only its new message | `ResidentPrefix.keep_end`, `end_offer` |
| prompt end | the state at the end of the last prompt, for a template that drops the last answer's reasoning (thinking on, no tools) | `ResidentPrefix.pend_offer` (SPD-215) |
| park and pin | another conversation's rows, swapped back at its next turn; the head of a prompt a long request replaced | `ResidentPrefix` `_park`, `park_offer`, `pin_offer` |
| state store, session | snapshots at grid lines and at a turn's end | engine/cache.py `StateStore`; server/app.py `_remember`, `_remember_turn` |
| spill | live conversations on the NVMe, read back in place | engine/spill.py `SessionSpill` |
| disk tier | prompt heads and chunks on the SSD, across restarts | engine/disk_cache.py `DiskTier`, server/diskcache.py |

A resumed prefill forwards at least one row. Each offer stops one row short of the prompt's end, so a draft head reads its picks again after any restore. A draft head that keeps other state across rounds must put it in the sections, or have that row rebuild it. Two capabilities change the guest rule for a family. With `chat_first`, a turn that holds the whole last prompt is never a guest of its own rows, and the conductor's question is a guest at any size. With `head_mark`, a prompt without a plan is marked where its first user turn starts (server/app.py `_head_mark`), so a repeated head is read once. The tokenizer then names the template's turn marks (`template_marks()`, read by `_marks_of`). `tests/test_adapter_guards.py` lists the families that set them, and `tests/test_resident_pinned.py` pins the other families' caching byte for byte.

Tests: `tests/test_cache_any_model.py` (a made-up state through every tier), `tests/test_end_anchor.py` (the same greedy tokens after a resume at an end), the family's own cache tests (a restore from every tier equals a cold prefill: `tests/test_hy4_serving.py`) and `tests/test_fp_v3.py` for its fingerprint.

### The conductor's helper calls

The conductor runs in TandemApp's extension and asks the served model short questions of its own: the class, the package. Each runs as a guest. With `chat_first` it runs over slot 0's chat at any size. It goes to an idle side slot when slot 0 holds a conversation of at least two chunks (server/app.py `_guest_of_slot0`). It keeps no session snapshot, and it gives back the rows it wrote in place (`ResidentPrefix.own_from`, `release_own`). With `head_mark` its head is cut at the head mark, so the state store holds it after the first call. How the router fits the model is `Capabilities.conductor`, a dict over the keys of TandemApp's `tandemapp/conductor/modelfit.py` `DEFAULTS`: `served_classifier`, `fallback`, `packages`, `no_package_classes`, `effort_notes`, `memory`, `rules` and `notes`. `notes: False` sets the four note keys off. A key TandemApp does not know stops its start, so a family adds only keys that exist. Flint and Hy4-flash declare `{"notes": False}`: their prompt has no place for the router's notes. LFM2.5 and NemotronH turn the served classifier off and fall back lean.

Tests: `tests/test_conductor_modelfit.py` (a declared fit reaches the router, as TandemApp resolves it), the family's question tests (the conductor's question beside a chat keeps the chat's end: `tests/test_hy4_serving.py`), and `tests/test_queue_wait.py` (a declared read rate seeds the prior).

### The effort policy

How a template takes an effort level is the family's `ReasoningSpec(effort=EffortSpec(words, default))` in its `arch.py` (engine/models/adapter.py). `words` maps each level to the word the template's `reasoning_effort` argument gets: the default is the shared `low`, `medium` and `xhigh`. Hy4-flash maps the levels `low`, `medium` and `high` to its mode `high`, and `xhigh` to `max`. `default` is the family's own level for a request nobody decided, the engine-alone default. Hy4-flash's is medium, with medium's 8,192-token cap and the answer reserve; its card states it (`effort_default`). The shared code does the rest. `server/api/thinking.py` `read` reads every request shape into one budget, and `build_prompt` (server/app.py) renders the word. In `_serve`, a client's number wins, a client's word gets its level's cap, and thinking off is the client's. A plan that decides the effort meets a client's word by the profile's `thinking.client_word`, in `routed()`: under `ceiling` the word caps the router's level, under `ignore` the router's level stands, and under `wins` the word is the turn's level with its cap. With nothing decided, the family's default level applies with its cap, else the server's `--reasoning-effort` word with no cap. `_effort_default_note` says at start when `--reasoning-effort` names a word such a family does not use. On the engine alone and under `wins`, an explicit effort is never capped below its own level. A family writes nothing when its template takes the shared words or ignores them; otherwise it writes one argument in `arch.py`.

Tests: `tests/test_effort_table.py` (every family's template call, prompt and budget for every effort a request can carry, against rows written on main, so a family's rows move only with its spec), `test_the_table_under_a_plan` and the `wins` tests in `tests/test_client_budgets.py`, `tests/test_modelspec.py` (the default spec is the shared words), and `tests/test_model_card.py` (a card's `effort_default` is its spec's).

## The tests a new model must pass

In this order, each on the new model:

1. `tools/refcheck.py`: argmax agreement with the published reference implementation on teacher-forced positions. The reference runs in its own process, because only one model fits on the board at a time. On Qwen3.8 the engine agrees on 99.25 % of 400 positions and on all 242 confident ones.
2. `tools/quality_gate.py` for every quantised weight set ([quantisation.md](quantisation.md)).
3. `tools/verify_spec.py`, the agreement gate with its near-tie rule, with every drafter ([exactness.md](exactness.md)).
4. The CPU suite, including `tests/test_refactor_gate.py`. That test records the exact bytes of a tiny model's run (every logit and every byte of state after a prefill, a decode, a block verify, a rollback, a tree verify and a commit) and fails if any byte moves. A change that is supposed to change arithmetic re-records the fixture and says why.
5. `ops/gate.sh` and the client smoke set before the numbers are published ([measurement.md](measurement.md)).

Flash-Next took the same gates in its own form, and a new model should take them too:

- `tests/test_load_shared.py`: the model starts through the one loader, and no model name, `hasattr(eng` or `isinstance(eng` appears under `server/`.
- `tests/test_cache_any_model.py` and the model's own cache test: a restore from every tier equals a cold prefill.
- `tests/test_fp_v3.py`: the model's fingerprint v3 moves with its kernels and tables and with nothing else, and the 27B's does not move.
- The load-time self-check: a recorded decode replayed as chains and trees of every verify width, every row bit for bit (3,936 rows on Flash-Next 3.05). A difference turns speculation off for the process and names the layer and the row.
- `tools/fn_spec_check.py` and `tools/fn_server_check.py --compare`: speculative greedy equals plain greedy, in the engine and through the server, and seeded sampled answers equal their plain twins (`--sampled`).

## What is still specific to Qwen3.8

- `engine/model.py` implements this family's forward: the Gated DeltaNet layer with its gated norm, attention with a gated output and partial rotary, and the dense MLP. A new family needs its own layer code behind the same verify, commit and cache interfaces. A model with a mixture of experts also needs expert routing inside the verify.
- The second family, Qwen3.8-Flash-Next (`engine/models/flashnext/`, turboderp's EXL3 packs), is served through the same server and request loop with its MTP drafter, the same caches, the KV pool and side slots ([server.md](server.md#qwen38-flash-next-srv-87), [caches.md](caches.md#what-a-state-is)); it reads images through its own tower, which its builder hands to the server's image path (`ctx.vision`, VIS-97).
- The recurrent-layer kernels assume Gated DeltaNet's state shape (heads of 128 by 128).
- Tile tables and router prices are measured for each model's shapes on the GB10. A new model or a new board needs its own sweep (`tools/fp8_probe.py --sweep`, `tools/nvfp4_skinny.py`, Flash-Next's `tools/fn_tiles.py`) and its own price file. A kernel registry keyed by op, weight format and device (ENG-251) and device profiles (ENG-255) are planned, not built.
- Some modules read their switches into globals at import. Two engines with different settings cannot share one process yet.
- Runtime names still carry the old project name, from the `QWEN38_` and `QSE_` variable prefixes to the `Qwen38Engine` class. A later release renames them.

## Register the model for the setup guide

The dashboard's setup guide (`#/setup`, [setup.md](setup.md)) lists one tile per `models/<id>/model.yaml`. The folder is named after the model's id, and the file follows schema 1: the weight profiles (one per `ops/profiles/<name>.env`), the measured table, the tools that run beside the model with their memory and serve.env keys, the parallelism limits, the methods and the device presets. [models/README.md](../models/README.md) lists the fields; `models/qwen3.8-27b/model.yaml` is the worked example, with the source of every number next to it.

```text
models/<id>/model.yaml                  the registry entry
models/<id>/measurements/<build>-*.json the raw data the measured table came from (optional)
models/schema/model.schema.json         the schema, for an editor (python -m server.setup.registry --schema)
```

Check a new file with `python -m server.setup.registry --check`. It refuses a file whose numbers do not add up (two recommended profiles, a tool that needs one that is off, a docs link that is not there), and `tests/test_setup_registry.py` fails when the file names a serve.env key nothing reads or when a profile in `ops/profiles/` has no entry. Memory is in GiB and comes from the engine's own `[memory] load stages` line and the dashboard's memory split, never from a guess.
