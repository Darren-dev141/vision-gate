#!/usr/bin/env python3
"""vision-gate test suite (plain asserts).

Run after `pip install .`:
    python tests/test_gate.py

Covers: identical -> drop; edited -> regions; localized small change;
ambiguous band -> fail-open pass; shape mismatch -> fail-open; sequence
keyframe selection; synthetic-video keyframe triage; receipts + stats
accounting; the `python -m vision_gate` entry point; and the optional
decision-endpoint hook (skipped, not failed, when no Jev-compatible
endpoint answers at $VG_DECISION_URL, default http://127.0.0.1:8765).

All fixtures are synthetic. No network access is required except for the
optional endpoint section.
"""
import json, os, subprocess, sys, tempfile

try:
    from vision_gate import gate
except ImportError:  # running from a source checkout without installing
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src"))
    from vision_gate import gate

import cv2
import numpy as np

FAIL = []
SKIP = []

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

def skip(name, why):
    print(f"SKIP {name}  ({why})")
    SKIP.append(name)

# --- fixtures: synthetic "screenshots" --------------------------------------
EDIT = (520, 420, 200, 80)   # the button/dialog area edited in variant b

def screen(variant):
    img = np.full((600, 800, 3), 32, np.uint8)
    cv2.rectangle(img, (0, 0), (800, 48), (52, 52, 60), -1)          # title bar
    cv2.putText(img, "Dashboard", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (230, 230, 230), 2)
    for i in range(6):                                                # text lines
        cv2.rectangle(img, (40, 90 + i * 42), (40 + 300 + (i * 37) % 180,
                      106 + i * 42), (150, 150, 150), -1)
    cv2.rectangle(img, (EDIT[0], EDIT[1]), (EDIT[0] + EDIT[2], EDIT[1] + EDIT[3]),
                  (70, 130, 200), -1)                                 # button
    cv2.putText(img, "Submit", (EDIT[0] + 55, EDIT[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if variant == "b":       # button area replaced by an alert dialog
        cv2.rectangle(img, (EDIT[0], EDIT[1]), (EDIT[0] + EDIT[2], EDIT[1] + EDIT[3]),
                      (200, 60, 50), -1)
        cv2.putText(img, "ERROR", (EDIT[0] + 55, EDIT[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if variant == "c":       # one small localized change, top-left
        cv2.rectangle(img, (100, 100), (124, 124), (255, 255, 255), -1)
    return img

def noisy(img, seed=7, frac=0.005):
    out = img.copy()
    rng = np.random.default_rng(seed)
    flat = out.reshape(-1, 3)
    idx = rng.choice(flat.shape[0], int(flat.shape[0] * frac), replace=False)
    flat[idx] = np.clip(flat[idx].astype(int) + 60, 0, 255).astype(np.uint8)
    return out

tmp = tempfile.mkdtemp(prefix="vg-test-")
def save(name, img):
    p = os.path.join(tmp, name)
    cv2.imwrite(p, img)
    return p

pa = save("a.png", screen("a"))
pb = save("b.png", screen("b"))
pc = save("c.png", screen("c"))
pn = save("n.png", noisy(screen("a")))
small = save("small.png", np.full((300, 400, 3), 32, np.uint8))

def run_cli(*argv, receipts=None, tag="test"):
    cmd = [sys.executable, "-m", "vision_gate", "--tag", tag]
    if receipts:
        cmd += ["--receipts", receipts]
    cmd += list(argv)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"CLI failed: {r.stderr}"
    return json.loads(r.stdout)

# --- 1. identical pair -> dropped, zero model frames -------------------------
v = run_cli("diff", pa, pa)
check("identical: not changed", v["changed"] is False)
check("identical: 0 frames to model", v["frames_passed_to_model"] == 0
      and v["frames_dropped"] == 1)
check("identical: no regions", v["regions"] == [])

# --- 2. edited pair -> changed, region lands on the edit ---------------------
v = run_cli("diff", pa, pb, "--annotate", os.path.join(tmp, "ann.png"))
check("edited: changed", v["changed"] is True and v["reason"] == "changed")
check("edited: 1 frame to model", v["frames_passed_to_model"] == 1)
cx, cy = EDIT[0] + EDIT[2] // 2, EDIT[1] + EDIT[3] // 2
top = v["regions"][0]
inter = (max(0, min(top["x"] + top["w"], EDIT[0] + EDIT[2]) - max(top["x"], EDIT[0])) *
         max(0, min(top["y"] + top["h"], EDIT[1] + EDIT[3]) - max(top["y"], EDIT[1])))
check("edited: top region covers the edit",
      inter >= 0.4 * EDIT[2] * EDIT[3] and top["x"] <= cx <= top["x"] + top["w"],
      f"region={top}")
check("edited: annotated image written",
      os.path.getsize(os.path.join(tmp, "ann.png")) > 0)

# --- 3. small localized change -> changed-localized ---------------------------
v = run_cli("diff", pa, pc)
check("localized: changed", v["changed"] is True
      and v["reason"] == "changed-localized", f"frac={v['changed_fraction']}")
check("localized: region near (100,100)",
      any(r["x"] <= 112 <= r["x"] + r["w"] and r["y"] <= 112 <= r["y"] + r["h"]
          for r in v["regions"]))

# --- 4. ambiguous band -> FAIL OPEN (passes to the model) ----------------------
v = run_cli("diff", pa, pn)
check("ambiguous: passes through (fail-open)",
      v["pass_through"] is True and v["frames_passed_to_model"] == 1,
      f"frac={v['changed_fraction']} reason={v['reason']}")
check("ambiguous: flagged ambiguous", v["ambiguous"] is True)

# --- 5. shape mismatch -> fail-open -------------------------------------------
v = run_cli("diff", pa, small)
check("shape mismatch: fail-open pass",
      v["pass_through"] is True and v["reason"] == "shape-mismatch-fail-open")

# --- 6. sequence: keyframes only where change happens --------------------------
seqdir = os.path.join(tmp, "seq"); os.mkdir(seqdir)
for i, var in enumerate(["a", "a", "a", "b", "b"]):
    cv2.imwrite(os.path.join(seqdir, f"f{i}.png"), screen(var))
v = run_cli("seq", "--dir", seqdir)
check("seq: keyframes [0, 3]", v["keyframes"] == [0, 3], f"got {v['keyframes']}")
check("seq: 2 passed / 3 dropped",
      v["frames_passed_to_model"] == 2 and v["frames_dropped"] == 3)

# --- 7. synthetic video: keyframe triage ---------------------------------------
# 90 frames at 320x240: static for 30, then a large moving block.
# MJPG is intra-frame only: identical input frames encode identically, so
# the static stretch diffs to exactly zero (no inter-frame codec noise).
vid = os.path.join(tmp, "synth.avi")
wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
for i in range(90):
    fr = np.full((240, 320, 3), 40, np.uint8)
    if i >= 30:
        x = 10 + (i - 30) * 4
        cv2.rectangle(fr, (x, 90), (x + 60, 150), (255, 255, 255), -1)
    wr.write(fr)
wr.release()
v = run_cli("video", vid, "--every", "3")
check("video: all sampled frames counted", v["frames_in"] == 30,
      f"frames_in={v['frames_in']}")
check("video: first sampled frame is a keyframe", v["keyframes"][0] == 0)
check("video: static stretch kept out", 5 not in v["keyframes"],
      f"keyframes={v['keyframes']}")
check("video: motion frames pass", 10 in v["keyframes"] and 29 in v["keyframes"],
      f"keyframes={v['keyframes']}")
check("video: counts add up",
      v["frames_passed_to_model"] == v["keyframe_count"]
      and v["frames_dropped"] == v["frames_in"] - v["keyframe_count"])

# --- 8. receipts + stats accounting --------------------------------------------
rp = os.path.join(tmp, "receipts.jsonl")
run_cli("diff", pa, pa, receipts=rp)
run_cli("diff", pa, pb, receipts=rp)
cmd = [sys.executable, "-m", "vision_gate", "--receipts", rp, "stats"]
r = subprocess.run(cmd, capture_output=True, text=True)
st = json.loads(r.stdout)
check("stats: totals add up",
      st["runs"] == 2 and st["frames_in"] == 2
      and st["frames_passed_to_model"] == 1 and st["frames_dropped"] == 1
      and st["kept_out_pct"] == 50.0, json.dumps(st))

# --- 9. library API import ------------------------------------------------------
gray_a = cv2.cvtColor(screen("a"), cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(screen("b"), cv2.COLOR_BGR2GRAY)
core = gate.compare_frames(gray_a, gray_b)
check("api: compare_frames importable and correct",
      core["changed"] is True and core["reason"] == "changed")

# --- 10. optional decision-endpoint hook (live endpoint only) -------------------
try:
    h = gate.endpoint_health()
    endpoint_up = h.get("status") == "ok"
except Exception:
    endpoint_up = False
if not endpoint_up:
    skip("endpoint: hook tests",
         f"no endpoint answering at {gate.DECISION_URL}")
else:
    v = run_cli("diff", pa, pb)
    res = gate.endpoint_decide(v, "Which region contains the alert dialog that "
                                  "replaced the Submit button?")
    check("endpoint: decision returned", bool(res.get("decision_id")),
          json.dumps(res)[:220])
    if res.get("decision_id"):
        recent = gate.endpoint_decisions(100)
        check("endpoint: decision_id in the endpoint's own log",
              res["decision_id"] in json.dumps(recent))
        check("endpoint: picked the edited region",
              res.get("region", {}).get("x", -1) <= cx <=
              res.get("region", {}).get("x", 0) + res.get("region", {}).get("w", 0))

print()
if SKIP:
    print(f"skipped: {SKIP}")
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("ALL TESTS PASSED")
