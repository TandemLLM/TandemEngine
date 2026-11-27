"""Hessian-calibrated EXL3 quantization (3.5 & 3.0) + K3 lm_head for AngelSlim/Hy4-flash-DFly2-B6 (OPS-341).

Runs in four composable stages on a single GPU (e.g. RTX PRO 6000 96 GB):
  --capture     Run target Hy4-flash EXL3 pack + BF16 DFly2-B6 on calibration rows; record exact
                FP32 Gram Hessians (H = X^T X) for every DFly projection & expert plus 8 held-out probes.
  --quant       Quantize DFly2-B6 projections to EXL3 (mul1 codebook) at --recipe 3.5 or 3.0.
  --quant-head  Quantize target lm_head to K=3 (mul1) using DFly2's captured pre-head Hessian.
  --verify      Load the quantized DFly2-B6 directory with `load_dfly_external(..., device="cuda")`
                and verify EXL3 MoEBank + AttentiveCorrection parity (>0.995 cosine) against probes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _acc_h(store: dict[str, torch.Tensor], cnt: dict[str, int], name: str, x: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1]).float()
    x2 = x2[torch.isfinite(x2).all(dim=1)]
    if x2.numel() == 0:
        return
    if name not in store:
        store[name] = torch.zeros(x2.shape[1], x2.shape[1], dtype=torch.float32, device=x2.device)
        cnt[name] = 0
    store[name].addmm_(x2.T, x2)
    cnt[name] += int(x2.shape[0])


def _load_calib_rows(path: str, seqs: int, min_len: int) -> list[list[int]]:
    d = load_file(path)
    ids = d["input_ids"]
    lens = d["lengths"]
    out: list[list[int]] = []
    for i in range(ids.shape[0]):
        n = int(lens[i].item())
        if n >= min_len:
            out.append(ids[i, :n].tolist())
        if len(out) >= seqs:
            break
    if len(out) < seqs:
        for i in range(ids.shape[0]):
            n = int(lens[i].item())
            if n > 32:
                row = ids[i, :n].tolist()
                while len(row) < min_len:
                    row = row + row
                out.append(row[:min_len])
            if len(out) >= seqs:
                break
    return out[:seqs]


@torch.no_grad()
def cmd_capture(args) -> None:
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import load, load_dfly_external

    os.makedirs(args.hess_dir, exist_ok=True)
    dev = args.device
    t0 = time.perf_counter()
    pack, W, cuts, tables, banks, _ = load(
        args.target, device=dev, table_policy="warm", splits=(dev == "cuda"), skip_dfly=True,
    )
    d_cfg, d_W, d_cuts, d_banks = load_dfly_external(args.dfly, pack.cfg, device=dev)
    pack.cfg.dfly = d_cfg
    cuts.update(d_cuts)

    eng = Hy4FlashEngine(pack.cfg, W, tables, dtype=torch.bfloat16, device=dev,
                         max_len=args.prompt + args.blocks * (d_cfg.draft_tokens + 1) + 64,
                         banks=banks, cuts=cuts)
    eng.dfly_cfg, eng.dfly_weights, eng.dfly_banks = d_cfg, d_W, d_banks
    eng.prefill_grouped = True
    eng.prefill_experts = "group"
    eng.prefill_v2 = dev == "cuda"
    if dev == "cuda":
        eng.bind_prompt_order()

    st = eng.new_state(eng.max_len)
    head = DFlyHead(type("CaptureSrv", (), {"fn": eng, "cfg": pack.cfg, "st": st, "graphs": None})(), graphs=False)

    # Map DFly projection tensors -> shared Hessian group key
    H: dict[str, torch.Tensor] = {}
    CNT: dict[str, int] = {}
    lin_owners: dict[int, str] = {}
    lin_owners[id(d_W["mtp_fusion.eh_proj.weight"])] = "mtp_fusion.eh_proj"
    for i in range(d_cfg.layers):
        p = f"layers.{i}."
        lin_owners[id(d_W[p + "self_attn.q_a_proj.weight"])] = f"l{i}.attn_in"
        lin_owners[id(d_W[p + "self_attn.linear_gate.weight"])] = f"l{i}.attn_in"
        lin_owners[id(d_W[p + "self_attn.q_b_proj.weight"])] = f"l{i}.q_b_proj"
        lin_owners[id(d_W[p + "self_attn.o_proj.weight"])] = f"l{i}.o_proj"
        lin_owners[id(d_W[p + "mlp.routed_expert_down_proj.weight"])] = f"l{i}.mlp_in"
        lin_owners[id(d_W[p + "mlp.shared_experts.gate_proj.weight"])] = f"l{i}.mlp_in"
        lin_owners[id(d_W[p + "mlp.shared_experts.up_proj.weight"])] = f"l{i}.mlp_in"
        lin_owners[id(d_W[p + "mlp.shared_experts.down_proj.weight"])] = f"l{i}.sh_down"
        lin_owners[id(d_W[p + "mlp.routed_expert_up_proj.weight"])] = f"l{i}.routed_up"
    if d_cfg.attentive_correction:
        lin_owners[id(d_W["hidden_correction.gate_proj.weight"])] = "hc.in"
        lin_owners[id(d_W["hidden_correction.up_proj.weight"])] = "hc.in"
        lin_owners[id(d_W["hidden_correction.attention.q_in.weight"])] = "hc.in"
        lin_owners[id(d_W["hidden_correction.down_proj.weight"])] = "hc.down"
        lin_owners[id(d_W["hidden_correction.attention.q_b_proj.weight"])] = "hc.q_b_proj"
        lin_owners[id(d_W["hidden_correction.attention.o_proj.weight"])] = "hc.o_proj"

    rms_owners: dict[int, str] = {}
    for i in range(d_cfg.layers):
        rms_owners[id(d_W[f"layers.{i}.self_attn.kv_a_layernorm.weight"])] = f"l{i}.kv_b_proj"
    if d_cfg.attentive_correction:
        rms_owners[id(d_W["hidden_correction.attention.kv_a_layernorm.weight"])] = "hc.kv_b_proj"

    exp_layers: dict[int, int] = {}
    for i in range(d_cfg.layers):
        exp_layers[id(d_W[f"layers.{i}.mlp.experts.gate_proj"])] = i

    lin_orig = eng.lin
    rms_orig = eng.rms
    exp_orig = eng.expert_pairs_bank
    head_orig = eng.head
    recording = False

    def wrap_lin(x, w):
        if recording:
            k = lin_owners.get(id(w))
            if k is not None:
                _acc_h(H, CNT, k, x)
        return lin_orig(x, w)

    def wrap_rms(x, w):
        out = rms_orig(x, w)
        if recording:
            k = rms_owners.get(id(w))
            if k is not None:
                _acc_h(H, CNT, k, out)
        return out

    def wrap_exp(z, idx, bank=None, experts=None, lim=None, grouped=False):
        if recording and experts is not None:
            li = exp_layers.get(id(experts[0]))
            if li is not None:
                G, U, _ = experts
                limit = eng.cfg.swiglu_limit if lim is None else lim
                for r in range(idx.shape[0]):
                    zr = z[r:r + 1]
                    for e in idx[r].tolist():
                        _acc_h(H, CNT, f"l{li}.e{e}.gu", zr)
                        g = F.linear(zr, G.weight(e, zr.dtype))
                        u = F.linear(zr, U.weight(e, zr.dtype))
                        if limit > 0:
                            g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
                        _acc_h(H, CNT, f"l{li}.e{e}.down", F.silu(g) * u)
        return exp_orig(z, idx, bank=bank, experts=experts, lim=lim, grouped=grouped)

    def wrap_head(n, w=None):
        if recording:
            _acc_h(H, CNT, "lm_head", n)
        return head_orig(n, w=w)

    eng.lin = wrap_lin
    eng.rms = wrap_rms
    eng.expert_pairs_bank = wrap_exp
    eng.head = wrap_head

    blk_size = d_cfg.draft_tokens + 1
    rows = _load_calib_rows(args.calib, args.seqs + args.probes, args.prompt + args.blocks * blk_size)
    calib_rows, probe_rows = rows[:args.seqs], rows[args.seqs:]
    probes = []

    for si, seq in enumerate(calib_rows + probe_rows):
        is_probe = si >= len(calib_rows)
        st.reset_inplace()
        st.pick_ok.zero_()
        prompt = seq[:args.prompt]
        tail = seq[args.prompt:]
        eng.prefill(torch.tensor(prompt, dtype=torch.int64), st, head="none")
        last_hid = st.prompt_hidden[-1:].clone()
        anchor = prompt[-1]
        for bi in range(args.blocks):
            blk = tail[bi * blk_size:(bi + 1) * blk_size]
            if len(blk) < blk_size:
                break
            anchor = blk[0]
            recording = not is_probe
            head.launch(last_hid, [anchor])
            toks, probs = head.read()
            recording = False
            if is_probe and bi == 0:
                head.stage(int(anchor), last_hid, st)
                probes.append({
                    "anchor": int(anchor),
                    "th": last_hid.detach().clone().cpu(),
                    "length": int(st.length),
                    "delta": int(st.delta),
                    "lat": {li: st.lat[li][:st.length].detach().clone().cpu() for li in sorted(set(d_cfg.kv_layers))},
                    "pick_layers": list(st.pick_layers),
                    "pick_q": st.pick_q.detach().clone().cpu(),
                    "pick_ok": st.pick_ok.detach().clone().cpu(),
                    "hidden": head._hidden(st).detach().clone().cpu(),
                    "tok": toks,
                    "prob": probs,
                })
            # Advance target state cleanly in 1 block call (teacher-forced calibration trajectory)
            parents = [-1] + list(range(len(blk) - 1))
            _, bh = eng.rows(blk, parents, st)
            eng.commit(st, list(range(len(blk))))
            last_hid = bh[-1:]
        if (si + 1) % 8 == 0 or si + 1 == len(calib_rows) + len(probe_rows):
            sample_k = next((k for k in ("l0.attn_in", "hc.in", "lm_head") if k in CNT), None)
            print(f"[capture] seq {si + 1}/{len(calib_rows) + len(probe_rows)} "
                  f"({time.perf_counter() - t0:.1f}s) "
                  f"{sample_k}:{CNT.get(sample_k, 0) if sample_k else 0}", flush=True)

    # Blend layer-mean Hessian for any rare or unvisited expert so LDL is strictly well-conditioned
    for i in range(d_cfg.layers):
        for suffix in ("gu", "down"):
            keys = [f"l{i}.e{e}.{suffix}" for e in range(d_cfg.num_experts) if f"l{i}.e{e}.{suffix}" in H]
            mean_h = torch.stack([H[k] / max(CNT[k], 1) for k in keys]).mean(dim=0)
            for e in range(d_cfg.num_experts):
                k = f"l{i}.e{e}.{suffix}"
                c = CNT.get(k, 0)
                if c < 64:
                    pad = 64 - c
                    H[k] = (H[k] if k in H else 0.0) + mean_h * pad
                    CNT[k] = c + pad

    for k, mat in H.items():
        torch.save({"H": mat.cpu(), "count": CNT[k]}, os.path.join(args.hess_dir, f"{k}.pt"))
    torch.save(probes, os.path.join(args.hess_dir, "probes.pt"))
    with open(os.path.join(args.hess_dir, "index.json"), "w") as f:
        json.dump({"target": args.target, "dfly": args.dfly, "seqs": args.seqs, "blocks": args.blocks,
                   "counts": {k: int(v) for k, v in sorted(CNT.items())}}, f, indent=1)
    print(f"[capture] saved {len(H)} Hessians + {len(probes)} probes -> {args.hess_dir} in {time.perf_counter() - t0:.1f}s")

    del head, st, eng, W, tables, banks, d_W, d_banks
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def _h_entry(cache: dict, hess_dir: str, name: str) -> dict:
    if name not in cache:
        d = torch.load(os.path.join(hess_dir, f"{name}.pt"), map_location="cpu")
        Hm = d["H"].cuda()
        cnt = int(d["count"])
        cache[name] = {
            "H": Hm, "first_key": name, "count": cnt, "finalized": False,
            "num_total": cnt * Hm.shape[0],
            "inf_nan": torch.zeros(2, dtype=torch.long, device="cuda"),
            "device": torch.device("cuda"),
        }
    return cache[name]


@torch.no_grad()
def cmd_quant(args) -> None:
    if args.exllamav3:
        sys.path.insert(0, args.exllamav3)
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from engine.config import resolve_snapshot

    snap = resolve_snapshot(args.dfly)
    with open(os.path.join(snap, "config.json")) as f:
        cfg_raw = json.load(f)
    nd = int(cfg_raw["num_hidden_layers"])
    E = int(cfg_raw["num_experts"])
    Im = int(cfg_raw["moe_intermediate_size"])
    Lz = int(cfg_raw.get("routed_expert_hidden_size") or cfg_raw["hidden_size"])

    idx_p = os.path.join(snap, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx_p))["weight_map"].values())) if os.path.isfile(idx_p) else ["model.safetensors"]
    w: dict[str, torch.Tensor] = {}
    for fn in files:
        with safe_open(os.path.join(snap, fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                canon = k[6:] if k.startswith("model.") else k
                w[canon] = f.get_tensor(k)

    # Bitrate schedules (everything 128-aligned -> zero padding; keep dense linears integer K=5 matching hy4 target packs)
    if args.recipe == "3.5":
        k_attn, k_hc, k_exp, k_shared = 5, 5, 3.5, 5
    elif args.recipe == "3.0":
        k_attn, k_hc, k_exp, k_shared = 5, 5, 3, 5
    else:
        raise ValueError(f"Unknown recipe {args.recipe!r}")

    jobs: list[tuple[str, str, int | float, torch.Tensor]] = [
        ("mtp_fusion.eh_proj", "mtp_fusion.eh_proj", k_hc, w["mtp_fusion.eh_proj.weight"]),
    ]
    for i in range(nd):
        s, m = f"layers.{i}.self_attn.", f"layers.{i}.mlp."
        jobs.extend([
            (s + "q_a_proj", f"l{i}.attn_in", k_attn, w[s + "q_a_proj.weight"]),
            (s + "linear_gate", f"l{i}.attn_in", k_attn, w[s + "linear_gate.weight"]),
            (s + "q_b_proj", f"l{i}.q_b_proj", k_attn, w[s + "q_b_proj.weight"]),
            (s + "kv_b_proj", f"l{i}.kv_b_proj", 5, w[s + "kv_b_proj.weight"]),
            (s + "o_proj", f"l{i}.o_proj", k_attn, w[s + "o_proj.weight"]),
            (m + "routed_expert_down_proj", f"l{i}.mlp_in", k_shared, w[m + "routed_expert_down_proj.weight"]),
            (m + "shared_experts.gate_proj", f"l{i}.mlp_in", k_shared, w[m + "shared_experts.gate_proj.weight"]),
            (m + "shared_experts.up_proj", f"l{i}.mlp_in", k_shared, w[m + "shared_experts.up_proj.weight"]),
            (m + "shared_experts.down_proj", f"l{i}.sh_down", k_shared, w[m + "shared_experts.down_proj.weight"]),
            (m + "routed_expert_up_proj", f"l{i}.routed_up", k_shared, w[m + "routed_expert_up_proj.weight"]),
        ])
        gu = w[m + "experts.gate_up_proj"]
        dn = w[m + "experts.down_proj"]
        assert gu.shape == (E, 2 * Im, Lz) and dn.shape == (E, Lz, Im), (
            f"Unexpected expert shapes at layer {i}: gu={tuple(gu.shape)} dn={tuple(dn.shape)}, "
            f"expected ({E}, {2 * Im}, {Lz}) and ({E}, {Lz}, {Im})"
        )
        for e in range(E):
            # AngelSlim exports [E, out, in] matching HYV4FlashExperts (gate first along dim 1).
            jobs.append((f"{m}experts.{e}.gate_proj", f"l{i}.e{e}.gu", k_exp, gu[e, :Im, :]))
            jobs.append((f"{m}experts.{e}.up_proj", f"l{i}.e{e}.gu", k_exp, gu[e, Im:, :]))
            jobs.append((f"{m}experts.{e}.down_proj", f"l{i}.e{e}.down", k_exp, dn[e]))

    if cfg_raw.get("attentive_correction"):
        hc = "hidden_correction."
        hca = hc + "attention."
        jobs.extend([
            (hc + "gate_proj", "hc.in", k_hc, w[hc + "gate_proj.weight"]),
            (hc + "up_proj", "hc.in", k_hc, w[hc + "up_proj.weight"]),
            (hca + "q_in", "hc.in", k_hc, w[hca + "q_in.weight"]),
            (hc + "down_proj", "hc.down", k_hc, w[hc + "down_proj.weight"]),
            (hca + "q_b_proj", "hc.q_b_proj", k_hc, w[hca + "q_b_proj.weight"]),
            (hca + "kv_b_proj", "hc.kv_b_proj", 5, w[hca + "kv_b_proj.weight"]),
            (hca + "o_proj", "hc.o_proj", k_hc, w[hca + "o_proj.weight"]),
        ])

    # Keep small/non-128 weights unquantized in original dtype
    quant_weights = {b + ".weight" for b, _, _, _ in jobs}
    out: dict[str, torch.Tensor] = {}
    for k, t in w.items():
        if ".mlp.experts." in k or k in quant_weights:
            continue
        out[k] = t.contiguous()

    hess_cache: dict[str, dict] = {}
    report: dict[str, dict] = {}
    t0 = time.perf_counter()
    for idx, (base, hname, K, mat) in enumerate(jobs):
        assert mat.shape[0] % 128 == 0 and mat.shape[1] % 128 == 0, f"{base} shape {tuple(mat.shape)} not 128-aligned"
        k_val = int(K) if float(K).is_integer() else float(K)
        qa = {"seed": idx, "K": k_val, "devices": [0], "apply_out_scales": None, "mul1": True}
        hd = _h_entry(hess_cache, args.hess_dir, hname)
        _, proxy_err, tensors = quantize_exl3(mat.float().T.contiguous().cuda(), hd, qa, False)
        for sub, t in tensors.items():
            if t is not None:
                out[f"{base}.{sub}"] = t.contiguous().cpu()
        report[base] = {"K": K, "proxy_err": float(proxy_err), "shape": list(mat.shape)}
        nxt_h = jobs[idx + 1][1] if idx + 1 < len(jobs) else None
        if hname != nxt_h:
            hess_cache.pop(hname, None)
        if ".experts." not in base or base.endswith((".experts.0.down_proj", ".experts.127.down_proj")):
            print(f"[quant {args.recipe}] {idx + 1}/{len(jobs)} {base:42s} K{K:g} err={float(proxy_err):.5f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    save_file(out, os.path.join(args.out_dir, "model.safetensors"),
              metadata={"source": args.dfly, "format": "exl3-mul1", "recipe": args.recipe})
    cfg_out = dict(cfg_raw)
    cfg_out["quantization_config"] = {
        "quant_method": "exl3", "format": "exl3-mul1", "bits": float(args.recipe),
        "codebook": "mul1", "expert_bits": k_exp, "attn_bits": k_attn, "hc_bits": k_hc,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(os.path.join(args.out_dir, "quant_report.json"), "w") as f:
        json.dump({"recipe": args.recipe, "elapsed_s": round(time.perf_counter() - t0, 1),
                   "mean_expert_proxy_err": sum(v["proxy_err"] for k, v in report.items() if ".experts." in k)
                   / max(1, sum(1 for k in report if ".experts." in k)),
                   "tensors": report}, f, indent=2)
    nb = sum(t.numel() * t.element_size() for t in out.values())
    print(f"[quant {args.recipe}] wrote {len(out)} tensors ({nb / 1e9:.3f} GB) -> {args.out_dir} in {time.perf_counter() - t0:.1f}s")


@torch.no_grad()
def cmd_quant_head(args) -> None:
    if args.exllamav3:
        sys.path.insert(0, args.exllamav3)
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(args.head_repo, "model.safetensors.index.json")
    wm = json.load(open(idx_path))["weight_map"]
    key = next((k for k in ("lm_head.weight", "model.lm_head.weight") if k in wm), None)
    if key is None:
        raise RuntimeError(f"lm_head.weight not in {args.head_repo}")
    shard_path = hf_hub_download(args.head_repo, wm[key])
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        W = f.get_tensor(key)

    d = torch.load(os.path.join(args.hess_dir, "lm_head.pt"), map_location="cpu")
    Hm = d["H"].cuda()
    cnt = int(d["count"])
    hd = {"H": Hm, "first_key": "lm_head", "count": cnt, "finalized": False,
          "num_total": cnt * Hm.shape[0], "inf_nan": torch.zeros(2, dtype=torch.long, device="cuda"),
          "device": torch.device("cuda")}
    k_val = int(args.k) if float(args.k).is_integer() else float(args.k)
    qa = {"seed": 0, "K": k_val, "devices": [0], "apply_out_scales": None, "mul1": True}
    t0 = time.perf_counter()
    _, proxy_err, tensors = quantize_exl3(W.float().T.contiguous().cuda(), hd, qa, False)
    out = {f"lm_head.{sub}": t.contiguous().cpu() for sub, t in tensors.items() if t is not None}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out, metadata={"source": args.head_repo, "format": "exl3-mul1", "K": str(args.k),
                                      "proxy_err": f"{float(proxy_err):.6f}"})
    nb = sum(t.numel() * t.element_size() for t in out.values())
    print(f"[quant-head] K{args.k:g} proxy_err={float(proxy_err):.6f} ({nb / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f}s -> {args.out}")


@torch.no_grad()
def cmd_verify(args) -> None:
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import load, load_dfly_external, load_draft_head_exl3

    dev = args.device
    probes = torch.load(os.path.join(args.hess_dir, "probes.pt"), map_location="cpu")
    pack, W, cuts, tables, banks, _ = load(
        args.target, device=dev, table_policy="warm", splits=(dev == "cuda"), skip_dfly=True,
    )
    d_cfg, d_W, d_cuts, d_banks = load_dfly_external(args.quant_dir, pack.cfg, device=dev)
    pack.cfg.dfly = d_cfg
    cuts.update(d_cuts)
    draft_head = None
    if args.head_file and os.path.isfile(args.head_file):
        draft_head, dh_cut = load_draft_head_exl3(args.head_file, pack.cfg, device=dev)
        if dh_cut is not None:
            cuts[id(draft_head)] = dh_cut
    eng = Hy4FlashEngine(pack.cfg, W, tables, dtype=torch.bfloat16, device=dev, max_len=4096, banks=banks, cuts=cuts)
    eng.dfly_cfg, eng.dfly_weights, eng.dfly_banks, eng.draft_head = d_cfg, d_W, d_banks, draft_head
    st = eng.new_state(4096)
    head = DFlyHead(type("VerifySrv", (), {"fn": eng, "cfg": pack.cfg, "st": st, "graphs": None})(), graphs=False)
    head_full = DFlyHead(type("VerifySrv2", (), {"fn": eng, "cfg": pack.cfg, "st": st, "graphs": None})(),
                         graphs=False, draft_head=eng.w("lm_head.weight"))

    match_full, match_dh, hidden_cos = 0, 0, []
    total_pos = 0
    for p in probes:
        st.length = p["length"]
        st.delta = p["delta"]
        st.picks_at = p["length"]
        for li, t in p["lat"].items():
            st.lat[li][:t.shape[0]].copy_(t.to(dev))
        st.pick_layers = list(p["pick_layers"])
        st.pick_q.copy_(p["pick_q"].to(dev))
        st.pick_ok.copy_(p["pick_ok"].to(dev))
        th = p["th"].to(dev)
        head_full.launch(th, [p["anchor"]])
        toks_f, _ = head_full.read()
        for j in range(head.depth):
            total_pos += 1
            if toks_f[j][0] == p["tok"][j][0]:
                match_full += 1
        if draft_head is not None:
            head.launch(th, [p["anchor"]])
            toks_d, _ = head.read()
            for j in range(head.depth):
                if toks_d[j][0] == p["tok"][j][0]:
                    match_dh += 1
        head_full.stage(p["anchor"], th, st)
        hq = head_full._hidden(st).float().reshape(1, -1)
        hb = p["hidden"].to(dev).float().reshape(1, -1)
        hidden_cos.append(float(F.cosine_similarity(hq, hb).item()))

    assert head_full.stats["stale"] == 0, f"full head went stale on probe: {head_full.stats}"
    if draft_head is not None:
        assert head.stats["stale"] == 0, f"draft head went stale on probe: {head.stats}"
    mean_cos = sum(hidden_cos) / len(hidden_cos)
    min_cos = min(hidden_cos)
    res = {
        "probes": len(probes), "positions": total_pos,
        "hidden_cos_mean": round(mean_cos, 6), "hidden_cos_min": round(min_cos, 6),
        "top1_match_full_head": round(match_full / max(1, total_pos), 4),
        "top1_match_k3_head": round(match_dh / max(1, total_pos), 4) if draft_head is not None else None,
    }
    print("[verify]", json.dumps(res), flush=True)
    with open(os.path.join(args.quant_dir, "verify_report.json"), "w") as f:
        json.dump(res, f, indent=2)
    assert mean_cos > 0.985, f"hidden cosine too low: {mean_cos}"


def _enable_sm120_profile() -> None:
    """Register an in-process RTX PRO 6000 (sm_120) GPU adapter so inline CUDA kernels compile for sm_120a."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        return
    import tools.gpu as _gpu
    from tools.gpu.base import _known
    from tools.gpu.gb10 import GB10

    class _SM120(GB10):
        id = "sm120_quant"
        about = "NVIDIA RTX PRO 6000 Blackwell (sm_120a) quantizer adapter"
        arches = ((12, 0),)
        names = ("RTX PRO 6000",)
        key = "rtx-pro-6000"
        unified_memory = False

        def arch_flags(self, kind: str) -> list[str]:
            _known(kind)
            return ["-gencode=arch=compute_120a,code=sm_120a"]

    _gpu.ADAPTERS = (_SM120,) + _gpu.ADAPTERS
    _gpu._BY_DEV.clear()
    _gpu._BY_KEY.clear()
    print(f"[gpu] enabled inline sm_120a profile: {_gpu.profile(0)}")


def main() -> None:
    _enable_sm120_profile()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("capture")
    p1.add_argument("--target", required=True)
    p1.add_argument("--dfly", required=True)
    p1.add_argument("--calib", required=True)
    p1.add_argument("--hess-dir", required=True)
    p1.add_argument("--seqs", type=int, default=80)
    p1.add_argument("--prompt", type=int, default=256)
    p1.add_argument("--blocks", type=int, default=16)
    p1.add_argument("--probes", type=int, default=8)
    p1.add_argument("--device", default="cuda")

    p2 = sub.add_parser("quant")
    p2.add_argument("--dfly", required=True)
    p2.add_argument("--hess-dir", required=True)
    p2.add_argument("--recipe", choices=("3.5", "3.0"), required=True)
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--exllamav3", default="")

    p3 = sub.add_parser("quant-head")
    p3.add_argument("--head-repo", default="tencent/Hy4-flash")
    p3.add_argument("--hess-dir", required=True)
    p3.add_argument("--k", type=float, default=3.0)
    p3.add_argument("--out", required=True)
    p3.add_argument("--exllamav3", default="")

    p4 = sub.add_parser("verify")
    p4.add_argument("--target", required=True)
    p4.add_argument("--dfly", default="")
    p4.add_argument("--quant-dir", required=True)
    p4.add_argument("--hess-dir", required=True)
    p4.add_argument("--head-file", default="")
    p4.add_argument("--device", default="cuda")

    a = ap.parse_args()
    {"capture": cmd_capture, "quant": cmd_quant, "quant-head": cmd_quant_head, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
