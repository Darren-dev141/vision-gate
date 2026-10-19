# vision-gate

**A fail-open change-detection front door for vision pipelines.** Cheap
frame differencing (OpenCV) decides whether anything visual actually
changed — and where — *before* any vision-model call is spent. The product
is the pass/drop decision with receipts, not CV speed: a differencing pass
over a short clip costs a fraction of a second, while the expensive step in
a vision pipeline is model attention. The gate rations that attention.

## The fail-open rule (read this first)

**A gate that drops real change is worse than no gate.** Dropping happens
only on positive evidence of *no* change. Everything else passes through:

| Situation | Verdict |
|---|---|
| changed fraction ≥ threshold (default 0.02) | pass — `changed` |
| a dense ("hot") region exists below the global threshold | pass — `changed-localized` |
| changed fraction in the ambiguous band (0.001–0.02, no hot region) | pass — `ambiguous-band-fail-open` |
| frames can't be aligned (shape mismatch) | pass — `shape-mismatch-fail-open` |
| changed fraction ≤ noise floor (0.001) and no real region | **drop** — `unchanged` |

Regions are connected components of changed pixels (specks under 0.05% of
the frame are noise), merged when their boxes come within 16 px. A region
is `hot` when ≥25% of its box actually changed.

## Install

```bash
pip install git+https://github.com/Darren-dev141/vision-gate
# or from a clone:
pip install .
```

Requires Python ≥3.10, numpy, opencv-python-headless. CPU-only; no cloud
calls, no uploads.

## CLI

```bash
vision-gate diff A.png B.png [--tol 25] [--thresh 0.02] [--min-region 0.0005]
                             [--ask "which region ...?"] [--annotate out.png]
vision-gate seq img1 img2 ...            # or: vision-gate seq --dir DIR
vision-gate video path.mp4 [--kf-thresh 0.02] [--every 1] [--scale 960x540|native]
vision-gate decide verdict.json --ask "which region ...?"
vision-gate stats
```

(Equivalently `python -m vision_gate ...`.) Every command prints exactly
one JSON verdict on stdout (exit 0; errors are JSON on stderr, exit 1):

```json
{"changed": true, "pass_through": true, "ambiguous": false,
 "reason": "changed", "changed_fraction": 0.0334,
 "regions": [{"x": 520, "y": 420, "w": 201, "h": 81,
              "area_fraction": 0.0339, "changed_fraction": 0.9837,
              "hot": true}],
 "mode": "diff", "inputs": [{"path": "...", "sha256": "..."}],
 "frames_in": 1, "frames_passed_to_model": 1, "frames_dropped": 0,
 "latency_ms": 41.2}
```

`seq` adds `keyframes` (frame 0 always passes; a later frame passes when
it differs from its predecessor). `video` samples a video and returns its
keyframes in the same shape.

## Receipts — the savings ledger

Every CLI run appends one JSONL record to `receipts.jsonl` (override with
`--receipts PATH` or `$VG_RECEIPTS`, skip with `--no-receipt`, label with
`--tag`/`$VG_TAG`): timestamp, mode, input paths + SHA-256, params, frames
in / passed to model / dropped, region count, latency, and the
`decision_id` when the hook fired. `vision-gate stats` totals the ledger,
including the kept-out percentage. (Kept-out is workload-dependent: idle
screens keep out almost everything; near-constant-motion video keeps out
very little — the 2% rule passes busy content, by design.)

## Optional: the decision-endpoint hook

When a verdict's regions need a *decision* — which region is the target,
which change is the one that matters — the gate can pose the candidate
regions (≤10, largest first, described in words: position, size, coverage,
density) to a decision endpoint and return its pick with confidence and a
`decision_id`:

```bash
vision-gate diff before.png after.png --ask "Which region contains the error dialog?"
# -> verdict + "decision": {"choice": "region_0", "confidence": 1.0,
#                           "decision_id": "14f15b1ed34f", "region": {...}}
```

The endpoint is **any Jev-compatible decision server** — one exposing
`GET /health`, `POST /choose` (`{state, options, instructions}` →
`{choice, confidence, probabilities, decision_id}`), and
`GET /decisions?limit=N`. Set its base URL with `$VG_DECISION_URL`
(default `http://127.0.0.1:8765`). The hook is optional and off the hot
path: plain diff/seq/video never call it, the decision lands in the
endpoint's own log (so calibration and outcome-recording work on it
unchanged), and if the endpoint is down, `--ask` returns an error object
in the verdict — the gate itself still answers. Verified live against a
local [Laya](https://github.com/NandhaKishorM/laya) decision server: test
decisions `319564e6a3f6` and `3e0b058c740f` and demo decision
`14f15b1ed34f` are present in that server's own decision log, with the
edited region picked correctly.

## Who consumes it

- **Screenshot automation** — poll with `diff` (or `seq` on a capture
  buffer); spend a vision-model call only when `pass_through` is true,
  and hand the model the regions, not the whole frame, when they exist.
- **Video/reel triage** — `video` mode is the front-end for
  media-understanding pipelines: keyframes are the frames worth model
  attention.
- **Agent tooling** — the CLI is the contract: one JSON verdict on
  stdout, exit codes, a receipts file as the shared ledger. The gate is a
  client-side layer; it changes nothing upstream.

## Measured results

All measured on the original build (CPU-only) with the test suite and
demo in this repo:

- **Idle screen, 10 frames → 1 passed to a model, 9 kept out.** An agent
  watching an unchanging screen spends one model call, not ten.
- **Alert appearing at frame 6 of 10 → keyframes exactly [0, 6].** The
  alert's diff measures **3.34% of pixels changed**, localized to a
  single hot region on the changed UI element.
- **Ambiguous band fails open.** Borderline change (between the noise
  floor and the pass threshold, no hot region) is passed to the model and
  flagged `ambiguous` — never dropped.
- **Video keyframe parity, bit-exact.** The gate's keyframe selection
  matched an independent baseline keyframe-triage implementation exactly
  on two real videos: **285/285** keyframes identical on one, **39/39**
  on the other.
- **Decision hook verified end-to-end** against a live endpoint (see the
  decision_ids above).

## Tests & demo

```bash
pip install .
python tests/test_gate.py   # plain asserts; the endpoint section skips
                            # cleanly when no endpoint is configured
python demo/demo.py         # synthetic fixtures only
```

The demo runs three scenarios: an idle screen (10 frames → 1 model
call), an alert at frame 6 (keyframes [0, 6], change localized, endpoint
pick when configured), and a synthetic video (static, then motion) under
keyframe triage.

## License

MIT — see [LICENSE](LICENSE).
