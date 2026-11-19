#!/bin/bash
# The Hy4-flash pack's gates on this box, in one bounded window of the box alone.
#
#   ops/hy4-gate.sh --uploaded [--repo ORG/NAME] [--revision BRANCH] [--dir PACK_DIR] [--out DIR]
#                   [--minutes 150] [--idle 600] [--deadline 7200] [--fit 262144] [--skip-download] [--dry-run]
#
# 1. download  the pack from its private repository into PACK_DIR (the live engine stays up). The token is the
#              library's own (~/.cache/huggingface/token); it is never read, printed or passed here.
# 2. audit     every file's size and sha256 against the repository's records, then the header audit
#              (engine/models/hy4_flash/pack.py: a missing tensor, a wrong shape, a truncated shard stop it by name).
#              Still beside the live engine, at nice 19.
# 3. idle      waits until the live engine has had no running and no waiting request for --idle seconds in a row
#              (default 600), at most --deadline seconds; no idle window, no hold.
# 4. hold      under the box lock, the live release's own ops/hold.sh stops the service, runs
#              tools/hy4_check.py (load, the self-check on the plain pass and through the graphs, the prompt's
#              experts, plain decode, drafted rounds, the fit run of --fit tokens) and restarts the service.
# 5. record    OUT/gate.log with the UTC start and end of every step, OUT/check.jsonl with the tool's lines,
#              and the live engine's /health after the hold.
#
# --uploaded says the pack's upload is finished (its notes say so); without it nothing but --dry-run runs.
# The engine tree is the one this script is in; HY4_PY names its interpreter (default: the live release's).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
REPO="0xBakeer/TandemLLM-Hy4-flash-EXL3"; REV="3.08bpw_h5_ng5"
DIR="$HOME/models/hy4-flash-exl3-3.08"; OUT=""
MIN=150; IDLE=600; DEADLINE=7200; FIT=262144; SKIP=0; DRY=0; OK=0
LOCK="${HY4_BOX_LOCK:-$HOME/.qwen38-box.flock}"; HEALTH="${HY4_HEALTH:-http://127.0.0.1:8000/health}"
NEED_GB=120
while [ $# -gt 0 ]; do
    case "$1" in
        --uploaded) OK=1 ;; --dry-run) DRY=1 ;; --skip-download) SKIP=1 ;;
        --repo) REPO="$2"; shift ;; --revision) REV="$2"; shift ;; --dir) DIR="$2"; shift ;; --out) OUT="$2"; shift ;;
        --minutes) MIN="$2"; shift ;; --idle) IDLE="$2"; shift ;; --deadline) DEADLINE="$2"; shift ;; --fit) FIT="$2"; shift ;;
        *) echo "[hy4-gate] unknown argument $1"; exit 2 ;;
    esac
    shift
done
# the floor of the idle wait; a knob only for the tests (a compressed clock)
IDLE_MIN="${HY4_GATE_IDLE_MIN:-600}"
[ "$IDLE" -ge "$IDLE_MIN" ] || { echo "[hy4-gate] REFUSING: the idle wait is at least $IDLE_MIN s"; exit 2; }
[ "$MIN" -le 180 ] || { echo "[hy4-gate] REFUSING: a window is at most 180 minutes"; exit 2; }
OUT="${OUT:-$HOME/hy4-gate/$(date -u +%Y%m%dT%H%MZ)}"
now() { date -u +%FT%TZ; }
say() { echo "$(now) $*" | tee -a "$OUT/gate.log"; }

live_tree() {
    local pid
    pid=$(ss -ltnpH 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && readlink -f "/proc/$pid/cwd"
}
LIVE="${HY4_LIVE:-$(live_tree)}"
PY="${HY4_PY:-${LIVE:+$LIVE/../venv/bin/python}}"
CHECK="$PY tools/hy4_check.py --pack $DIR --max-len 8192 --prompt 4096 --steps 128 --dfly --table cold --fit $FIT"
HOLD="flock $LOCK bash $LIVE/ops/hold.sh $MIN -- env -C $TREE PYTHONDONTWRITEBYTECODE=1 $CHECK"

if [ "$DRY" = 1 ]; then
    echo "[hy4-gate] plan (nothing runs):"
    echo "  tree      $TREE"
    echo "  live      ${LIVE:-<no engine on :8000>}"
    echo "  download  $REPO@$REV -> $DIR (needs $NEED_GB GB free)$( [ "$SKIP" = 1 ] && echo ' [skipped]')"
    echo "  audit     sizes and sha256 of every file, then the header audit"
    echo "  idle      $IDLE s with no running or waiting request, deadline $DEADLINE s"
    echo "  hold      $HOLD"
    echo "  record    $OUT"
    exit 0
fi
[ "$OK" = 1 ] || { echo "[hy4-gate] REFUSING: run with --uploaded once the pack's upload is finished"; exit 2; }
case "$LIVE" in */releases/*/src) ;; *) echo "[hy4-gate] REFUSING: the live engine is no release tree (${LIVE:-none})"; exit 3 ;; esac
[ -x "$PY" ] || { echo "[hy4-gate] REFUSING: no interpreter at $PY (HY4_PY)"; exit 3; }
mkdir -p "$OUT" "$DIR" || exit 3
say "START tree $TREE ($(git -C "$TREE" rev-parse --short HEAD 2>/dev/null || echo archive)), live $LIVE, pack $REPO@$REV -> $DIR"

if [ "$SKIP" = 0 ]; then
    FREE=$(df -BG --output=avail "$DIR" | tail -1 | tr -dc 0-9)
    HAVE=$(du -s -BG "$DIR" 2>/dev/null | cut -f1 | tr -dc 0-9)
    [ $(( FREE + ${HAVE:-0} )) -ge "$NEED_GB" ] || { say "REFUSING: $FREE GB free at $DIR, the pack needs $NEED_GB"; exit 3; }
    say "download: start"
    HF_XET_NUM_CONCURRENT_RANGE_GETS=64 nice -n 19 "$PY" - "$REPO" "$REV" "$DIR" <<'PYEOF' >> "$OUT/gate.log" 2>&1 || { say "download FAILED"; exit 4; }
import json, os, sys
from huggingface_hub import HfApi, snapshot_download
repo, rev, out = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=out)
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
want = {s.rfilename: {"size": s.size, "sha256": getattr(s.lfs, "sha256", None) if s.lfs else None} for s in info.siblings}
json.dump({"repo": repo, "revision": rev, "sha": info.sha, "files": want}, open(os.path.join(out, ".hy4-files.json"), "w"), indent=1)
print(f"downloaded {len(want)} files of {repo}@{info.sha}")
PYEOF
    say "download: done"
fi

say "audit: start"
nice -n 19 env -C "$TREE" PYTHONDONTWRITEBYTECODE=1 "$PY" - "$DIR" <<'PYEOF' >> "$OUT/gate.log" 2>&1 || { say "audit FAILED (see gate.log)"; exit 5; }
import hashlib, json, os, sys
sys.path.insert(0, ".")
d = sys.argv[1]
rec = os.path.join(d, ".hy4-files.json")
if os.path.isfile(rec):
    want = json.load(open(rec))["files"]
    for name, w in sorted(want.items()):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            sys.exit(f"missing file {name}")
        if w["size"] is not None and os.path.getsize(p) != w["size"]:
            sys.exit(f"{name}: {os.path.getsize(p)} bytes, the repository says {w['size']}")
        if w["sha256"]:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            if h.hexdigest() != w["sha256"]:
                sys.exit(f"{name}: sha256 {h.hexdigest()}, the repository says {w['sha256']}")
    print(f"sizes and sha256 of {len(want)} files equal the repository's")
else:
    print("no repository record beside the pack (--skip-download): sizes and sha256 not compared")
from engine.models.hy4_flash.pack import Hy4Pack
from engine.models.hy4_flash.table import EngramTable, table_file
pack = Hy4Pack.open(d)
print("header audit:", "; ".join(pack.audit()))
pack.check_mul1()
for li in pack.cfg.engram_layers:
    t = EngramTable(table_file(d, li), li)
    print(f"table {li}: {t.rows:,} rows at K {t.K}")
print(f"{len(pack.tensors)} tensors, {pack.resident_bytes() / (1 << 30):.2f} GiB of trunk and draft weights; DFly layers: {pack.has_dfly}")
PYEOF
say "audit: done"

say "idle: waiting for $IDLE s with no running or waiting request (deadline $DEADLINE s)"
start=""; deadline=$(( $(date +%s) + DEADLINE ))
while :; do
    t=$(date +%s)
    [ "$t" -gt "$deadline" ] && { say "no idle window of $IDLE s in $DEADLINE s: no hold"; exit 6; }
    busy=$(curl -s -m 5 "$HEALTH" | python3 -c "import json,sys; f=json.load(sys.stdin)['inflight']; print(f['running']+f['waiting'])" 2>/dev/null || echo NA)
    if [ "$busy" = 0 ]; then [ -z "$start" ] && start=$t; else start=""; fi
    [ -n "$start" ] && [ $(( t - start )) -ge "$IDLE" ] && break
    sleep "${HY4_GATE_POLL:-20}"
done
say "hold: start (budget $MIN min)"
HOLD_REASON="Hy4-flash pack gates: self-check, decode rates, a $FIT-token fit run (up to $MIN min)" \
    flock "$LOCK" bash "$LIVE/ops/hold.sh" "$MIN" -- env -C "$TREE" PYTHONDONTWRITEBYTECODE=1 $CHECK 2>&1 | tee "$OUT/hold.out" | grep -E '^\{"check"|^\[hold\]|^\[hy4\]|Error|Traceback' | tee -a "$OUT/gate.log"
rc=${PIPESTATUS[0]}
grep -E '^\{"check"' "$OUT/hold.out" > "$OUT/check.jsonl"
say "hold: done rc=$rc; live /health $(curl -s -o /dev/null -w '%{http_code}' -m 10 "$HEALTH")"
exit "$rc"
