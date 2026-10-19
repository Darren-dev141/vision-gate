#!/usr/bin/env python3
"""vision-gate: a fail-open change-detection front door for vision models.

Cheap frame differencing (OpenCV) decides whether anything visual actually
changed, and where, BEFORE any vision-model call is spent. The product is
the pass/drop decision with receipts, not CV speed: a full differencing
pass over a short clip costs a fraction of a second inside a pipeline whose
expensive step is model attention. The gate rations that attention.

FAIL-OPEN RULE: a gate that drops real change is worse than no gate.
Dropping happens only on positive evidence of *no* change. Everything
else -- the in-between change band, unreadable input, mismatched shapes --
passes the frame through to the model.

Usage:
    vision-gate diff A.png B.png [--tol 25] [--thresh 0.02]
                                 [--ask "which region ...?"] [--annotate out.png]
    vision-gate seq img1 img2 ...            (or: seq --dir DIR)
    vision-gate video path.mp4 [--kf-thresh 0.02] [--every 1]
    vision-gate decide verdict.json --ask "which region ...?"
    vision-gate stats

(Equivalently: python -m vision_gate ...)

Every command prints one JSON verdict on stdout. Exit 0 on success
(including "no change"), 1 on usage/IO errors (JSON error on stderr).
Every run appends a receipt to receipts.jsonl (override with --receipts or
$VG_RECEIPTS, skip with --no-receipt, label with --tag/$VG_TAG): inputs
hashed, frames in / passed to model / dropped, regions, latency, and the
decision_id when the optional decision-endpoint hook fired. `stats`
totals the savings ledger.

The --ask hook is OPTIONAL and off the hot path: it poses the verdict's
candidate regions to any Jev-compatible decision endpoint (POST /choose)
at $VG_DECISION_URL (default http://127.0.0.1:8765) and returns its pick.
Plain diff/seq/video never call the endpoint.

Requires opencv + numpy (installed as dependencies).
"""
import argparse, hashlib, json, os, sys, time, urllib.request
from datetime import datetime, timezone

try:
    import cv2
    import numpy as np
except ImportError:
    sys.stderr.write('{"error": "opencv/numpy missing; install with '
                     '`pip install vision-gate` or add opencv-python-headless '
                     'and numpy"}\n')
    raise SystemExit(1)

DEFAULT_RECEIPTS = os.environ.get("VG_RECEIPTS", "receipts.jsonl")
DECISION_URL = os.environ.get("VG_DECISION_URL", "http://127.0.0.1:8765")

# Defaults: tol 25 (per-pixel delta), keyframe fraction 0.02.
DEF_TOL = 25.0
DEF_THRESH = 0.02        # global changed-fraction at/above which a frame passes
DEF_NOISE = 0.001        # at/below this, with no real region, a frame is unchanged
DEF_MIN_REGION = 0.0005  # smallest region, as a fraction of frame area
DEF_FILL_HOT = 0.25      # a region is "hot" when this much of its box changed
DEF_MERGE_GAP = 16       # px: merge changed components this close together


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def _regions(mask, min_region_frac=DEF_MIN_REGION, merge_gap=DEF_MERGE_GAP):
    """Bounding-box regions of change from a boolean mask.

    Connected components of changed pixels (specks below min_region_frac
    of the frame are noise and dropped), then components whose boxes come
    within merge_gap px are merged into one region. A region is "hot" when
    at least DEF_FILL_HOT of its box actually changed -- a dense, real edit
    as opposed to scattered specks.
    """
    h, w = mask.shape
    frame_area = h * w
    min_area = max(8, int(min_region_frac * frame_area))
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    boxes = []  # [x0, y0, x1, y1, changed_px]
    for c in range(1, ncomp):
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[c, cv2.CC_STAT_LEFT]); y = int(stats[c, cv2.CC_STAT_TOP])
        boxes.append([x, y, x + int(stats[c, cv2.CC_STAT_WIDTH]),
                      y + int(stats[c, cv2.CC_STAT_HEIGHT]), area])
    # Merge boxes that (expanded by merge_gap) intersect, to a fixpoint.
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            hit = None
            for o in out:
                if not (b[0] - merge_gap > o[2] or o[0] - merge_gap > b[2]
                        or b[1] - merge_gap > o[3] or o[1] - merge_gap > b[3]):
                    hit = o
                    break
            if hit is None:
                out.append(b)
            else:
                hit[0] = min(hit[0], b[0]); hit[1] = min(hit[1], b[1])
                hit[2] = max(hit[2], b[2]); hit[3] = max(hit[3], b[3])
                hit[4] += b[4]
                merged = True
        boxes = out
    regions = []
    for x0, y0, x1, y1, _area in boxes:
        box_area = (x1 - x0) * (y1 - y0)
        changed_px = int(mask[y0:y1, x0:x1].sum())
        fill = changed_px / box_area if box_area else 0.0
        regions.append({
            "x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
            "area_fraction": round(box_area / frame_area, 6),
            "changed_fraction": round(float(fill), 6),
            "hot": bool(fill >= DEF_FILL_HOT),
        })
    regions.sort(key=lambda r: -r["area_fraction"])
    return regions


def compare_frames(a, b, tol=DEF_TOL, thresh=DEF_THRESH, noise=DEF_NOISE,
                   min_region=DEF_MIN_REGION):
    """Compare two grayscale frames. Returns the verdict core (no I/O fields).

    Decision order (fail-open):
      shapes differ                    -> pass (ambiguous: shape-mismatch)
      global fraction >= thresh        -> pass (changed)
      a hot region exists              -> pass (changed, localized)
      global fraction <= noise         -> DROP (unchanged)
      anything in between              -> pass (ambiguous: fail-open)
    """
    if a.shape != b.shape:
        return {"changed": True, "pass_through": True, "ambiguous": True,
                "reason": "shape-mismatch-fail-open", "changed_fraction": None,
                "regions": []}
    d = cv2.absdiff(a, b)
    _, bw = cv2.threshold(d, tol, 255, cv2.THRESH_BINARY)
    mask = bw > 0
    frac = float(mask.mean())
    regions = _regions(mask, min_region)
    hot = any(r["hot"] for r in regions)

    base = {"changed_fraction": round(frac, 6), "regions": regions}
    if frac >= thresh:
        return {**base, "changed": True, "pass_through": True,
                "ambiguous": False, "reason": "changed"}
    if hot:
        return {**base, "changed": True, "pass_through": True,
                "ambiguous": False, "reason": "changed-localized"}
    if frac <= noise:
        return {**base, "changed": False, "pass_through": False,
                "ambiguous": False, "reason": "unchanged"}
    return {**base, "changed": True, "pass_through": True,
            "ambiguous": True, "reason": "ambiguous-band-fail-open"}


def _finish(verdict, mode, inputs, params, t0, frames_in, frames_passed):
    verdict.update({
        "mode": mode,
        "inputs": inputs,
        "params": params,
        "frames_in": frames_in,
        "frames_passed_to_model": frames_passed,
        "frames_dropped": frames_in - frames_passed,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    })
    return verdict


def write_receipt(verdict, path, tag):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "mode": verdict["mode"],
        "inputs": verdict["inputs"],
        "params": verdict["params"],
        "changed": verdict.get("changed"),
        "frames_in": verdict["frames_in"],
        "frames_passed_to_model": verdict["frames_passed_to_model"],
        "frames_dropped": verdict["frames_dropped"],
        "regions": len(verdict.get("regions", [])),
        "latency_ms": verdict["latency_ms"],
        "decision_id": (verdict.get("decision") or {}).get("decision_id"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# --- optional decision-endpoint hook -----------------------------------------
# Something else sees (this gate); a decision endpoint decides. The endpoint
# is any Jev-compatible server exposing:
#   GET  /health                  -> {"status": "ok", ...}
#   POST /choose {state, options, instructions}
#                                 -> {"choice", "confidence", "probabilities",
#                                     "decision_id"}
#   GET  /decisions?limit=N       -> recent decision records (for auditing)
# (Laya is one such endpoint.) The hook is off the hot path: plain
# diff/seq/video never call it, and if the endpoint is unreachable the
# verdict carries an error object -- the gate itself still answers.

def _endpoint_get(path, timeout=10):
    with urllib.request.urlopen(DECISION_URL + path, timeout=timeout) as resp:
        return json.loads(resp.read())


def _endpoint_choose(state, options, instructions):
    req = urllib.request.Request(
        DECISION_URL + "/choose",
        data=json.dumps({"state": state, "options": options,
                         "instructions": instructions}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def endpoint_health():
    return _endpoint_get("/health")


def endpoint_decisions(limit=50):
    return _endpoint_get(f"/decisions?limit={limit}")


def region_options(regions, limit=10):
    """Candidate regions as decision-endpoint options (<=10, largest first)."""
    opts = {}
    for i, r in enumerate(regions[:limit]):
        opts[f"region_{i}"] = (
            f"region {i}: box at x={r['x']}, y={r['y']}, size "
            f"{r['w']}x{r['h']} px, covers {r['area_fraction'] * 100:.2f}% of "
            f"the frame, {r['changed_fraction'] * 100:.1f}% of its pixels changed")
    return opts


def endpoint_decide(verdict, question):
    """Pose the verdict's candidate regions to the decision endpoint."""
    regions = verdict.get("regions") or []
    if not regions:
        return {"error": "no regions to decide over"}
    try:
        health = endpoint_health()
        if health.get("status") != "ok":
            return {"error": f"decision endpoint health not ok: {health}"}
        opts = region_options(regions)
        state = (f"A vision gate compared two frames: "
                 f"{(verdict.get('changed_fraction') or 0) * 100:.2f}% of pixels "
                 f"changed across {len(regions)} region(s). {question}")
        res = _endpoint_choose(state, opts,
                               "Which region best answers the question?")
    except Exception as e:
        return {"error": f"decision endpoint unreachable: {e}"}
    choice = res.get("choice")
    out = {"choice": choice, "confidence": res.get("confidence"),
           "probabilities": res.get("probabilities"),
           "decision_id": res.get("decision_id")}
    if choice in opts:
        out["region_index"] = int(choice.split("_")[1])
        out["region"] = regions[out["region_index"]]
    return out


# --- modes -------------------------------------------------------------------

def mode_diff(args):
    t0 = time.time()
    a, b = load_gray(args.a), load_gray(args.b)
    v = compare_frames(a, b, args.tol, args.thresh, args.noise, args.min_region)
    inputs = [{"path": os.path.abspath(p), "sha256": sha256_file(p)}
              for p in (args.a, args.b)]
    passed = 1 if v["pass_through"] else 0
    v = _finish(v, "diff", inputs,
                {"tol": args.tol, "thresh": args.thresh, "min_region": args.min_region},
                t0, frames_in=1, frames_passed=passed)
    if args.annotate:
        img = cv2.imread(args.b)
        for r in v["regions"]:
            cv2.rectangle(img, (r["x"], r["y"]),
                          (r["x"] + r["w"], r["y"] + r["h"]), (0, 0, 255), 2)
        cv2.imwrite(args.annotate, img)
        v["annotated"] = os.path.abspath(args.annotate)
    if args.ask:
        v["decision"] = endpoint_decide(v, args.ask)
    return v


def mode_seq(args):
    t0 = time.time()
    paths = list(args.images)
    if args.dir:
        import glob
        paths += sorted(glob.glob(os.path.join(args.dir, "*.png")) +
                        glob.glob(os.path.join(args.dir, "*.jpg")))
    if len(paths) < 2:
        raise ValueError("seq needs at least 2 images")
    frames = [load_gray(p) for p in paths]
    keyframes, pairs = [0], []
    for i in range(1, len(frames)):
        pv = compare_frames(frames[i - 1], frames[i], args.tol, args.thresh,
                            args.noise, args.min_region)
        pairs.append({"frame": i, "changed": pv["changed"],
                      "pass_through": pv["pass_through"],
                      "reason": pv["reason"],
                      "changed_fraction": pv["changed_fraction"]})
        if pv["pass_through"]:
            keyframes.append(i)
    v = {"changed": any(p["changed"] for p in pairs),
         "pass_through": len(keyframes) > 1,
         "reason": "sequence", "keyframes": keyframes, "pairs": pairs,
         "regions": []}
    inputs = [{"path": os.path.abspath(p), "sha256": sha256_file(p)}
              for p in paths]
    return _finish(v, "seq", inputs,
                   {"tol": args.tol, "thresh": args.thresh, "min_region": args.min_region},
                   t0, frames_in=len(frames), frames_passed=len(keyframes))


def mode_video(args):
    t0 = time.time()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {args.video}")
    cap_w = cap_h = None
    if args.scale and args.scale != "native":
        cap_w, cap_h = (int(s) for s in args.scale.lower().split("x"))
    idx = sampled = 0
    prev = None
    keyframes = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cap_w and (gray.shape[1] > cap_w or gray.shape[0] > cap_h):
                f = min(cap_w / gray.shape[1], cap_h / gray.shape[0])
                gray = cv2.resize(gray, (int(gray.shape[1] * f),
                                         int(gray.shape[0] * f)))
            height, width = gray.shape
            if prev is None:
                keyframes.append(sampled)
            else:
                pv = compare_frames(prev, gray, args.tol, args.kf_thresh,
                                    args.noise, args.min_region)
                if pv["pass_through"]:
                    keyframes.append(sampled)
            prev = gray
            sampled += 1
        idx += 1
    cap.release()
    if sampled == 0:
        raise ValueError(f"no frames decoded from video: {args.video}")
    v = {"changed": len(keyframes) > 1, "pass_through": True,
         "reason": "video-keyframe-triage", "regions": [],
         "keyframes": keyframes,
         "keyframe_count": len(keyframes),
         "width": width, "height": height}
    inputs = [{"path": os.path.abspath(args.video),
               "sha256": sha256_file(args.video)}]
    return _finish(v, "video", inputs,
                   {"tol": args.tol, "kf_thresh": args.kf_thresh,
                    "every": args.every, "scale": args.scale},
                   t0, frames_in=sampled, frames_passed=len(keyframes))


def mode_decide(args):
    with open(args.verdict) as f:
        v = json.load(f)
    return {"verdict_mode": v.get("mode"), "question": args.ask,
            "decision": endpoint_decide(v, args.ask)}


def mode_stats(args):
    path = args.receipts or DEFAULT_RECEIPTS
    tot = {"runs": 0, "frames_in": 0, "frames_passed_to_model": 0,
           "frames_dropped": 0, "endpoint_decisions": 0, "by_mode": {}}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tot["runs"] += 1
                for k in ("frames_in", "frames_passed_to_model", "frames_dropped"):
                    tot[k] += r.get(k, 0)
                if r.get("decision_id"):
                    tot["endpoint_decisions"] += 1
                m = tot["by_mode"].setdefault(r["mode"], {"runs": 0, "frames_in": 0,
                                              "frames_dropped": 0})
                m["runs"] += 1
                m["frames_in"] += r.get("frames_in", 0)
                m["frames_dropped"] += r.get("frames_dropped", 0)
    tot["kept_out_pct"] = (round(100.0 * tot["frames_dropped"] / tot["frames_in"], 2)
                           if tot["frames_in"] else 0.0)
    tot["receipts"] = path
    return tot


def main():
    ap = argparse.ArgumentParser(prog="vision-gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipts", default=None, help="receipts JSONL path")
    ap.add_argument("--no-receipt", action="store_true")
    ap.add_argument("--tag", default=os.environ.get("VG_TAG", "cli"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--tol", type=float, default=DEF_TOL)
        p.add_argument("--thresh", type=float, default=DEF_THRESH)
        p.add_argument("--noise", type=float, default=DEF_NOISE)
        p.add_argument("--min-region", type=float, default=DEF_MIN_REGION)

    p = sub.add_parser("diff"); common(p)
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--ask", default=None,
                   help="pose regions to the decision endpoint with this question")
    p.add_argument("--annotate", default=None, help="write boxed image here")
    p.set_defaults(fn=mode_diff)

    p = sub.add_parser("seq"); common(p)
    p.add_argument("images", nargs="*")
    p.add_argument("--dir", default=None)
    p.set_defaults(fn=mode_seq)

    p = sub.add_parser("video")
    p.add_argument("video")
    p.add_argument("--tol", type=float, default=DEF_TOL)
    p.add_argument("--noise", type=float, default=DEF_NOISE)
    p.add_argument("--min-region", type=float, default=DEF_MIN_REGION)
    p.add_argument("--kf-thresh", type=float, default=DEF_THRESH)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--scale", default="960x540",
                   help="cap frame size for triage (default 960x540; "
                        "'native' for full resolution)")
    p.set_defaults(fn=mode_video)

    p = sub.add_parser("decide")
    p.add_argument("verdict", help="a verdict JSON printed by diff/seq")
    p.add_argument("--ask", required=True)
    p.set_defaults(fn=mode_decide, no_receipt=True)

    p = sub.add_parser("stats")
    p.set_defaults(fn=mode_stats, no_receipt=True)

    args = ap.parse_args()
    try:
        v = args.fn(args)
    except Exception as e:  # fail-open on tool error too: say so, exit 1
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    if not getattr(args, "no_receipt", False):
        write_receipt(v, args.receipts or DEFAULT_RECEIPTS, args.tag)
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
