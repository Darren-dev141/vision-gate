#!/usr/bin/env python3
"""vision-gate end-to-end demo (synthetic fixtures only).

Scenario 1: an agent watches a screen that does NOT change (10 frames) --
the gate spends 1 model frame instead of 10.
Scenario 2: an alert appears at frame 6 -- only frames 0 and 6 pass, the
gate localizes the alert, and (if a decision endpoint is configured) the
endpoint picks the region that matters.
Scenario 3: a synthetic video -- static for a second, then motion --
keyframe triage counts.

Run after `pip install .`:
    python demo/demo.py
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

EDIT = (520, 420, 200, 80)

def screen(alert=False):
    img = np.full((600, 800, 3), 32, np.uint8)
    cv2.rectangle(img, (0, 0), (800, 48), (52, 52, 60), -1)
    cv2.putText(img, "Dashboard", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (230, 230, 230), 2)
    for i in range(6):
        cv2.rectangle(img, (40, 90 + i * 42), (40 + 300 + (i * 37) % 180,
                      106 + i * 42), (150, 150, 150), -1)
    color = (200, 60, 50) if alert else (70, 130, 200)
    label = "ERROR" if alert else "Submit"
    cv2.rectangle(img, (EDIT[0], EDIT[1]), (EDIT[0] + EDIT[2], EDIT[1] + EDIT[3]),
                  color, -1)
    cv2.putText(img, label, (EDIT[0] + 55, EDIT[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

tmp = tempfile.mkdtemp(prefix="vg-demo-")
RECEIPTS = os.path.join(tmp, "receipts.jsonl")

def run_gate(*argv):
    cmd = [sys.executable, "-m", "vision_gate", "--tag", "demo",
           "--receipts", RECEIPTS] + list(argv)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)

total_in = total_passed = 0

print("== Scenario 1: watching an idle screen (10 identical frames) ==")
d1 = os.path.join(tmp, "idle"); os.mkdir(d1)
for i in range(10):
    cv2.imwrite(os.path.join(d1, f"f{i:02d}.png"), screen())
v = run_gate("seq", "--dir", d1)
print(f"frames in: {v['frames_in']}  passed to model: "
      f"{v['frames_passed_to_model']}  dropped: {v['frames_dropped']}")
print(f"naive pipeline: 10 model calls -> gated: "
      f"{v['frames_passed_to_model']} model call(s)")
assert v["keyframes"] == [0]
total_in += v["frames_in"]; total_passed += v["frames_passed_to_model"]

print("\n== Scenario 2: an alert appears at frame 6 ==")
d2 = os.path.join(tmp, "alert"); os.mkdir(d2)
for i in range(10):
    cv2.imwrite(os.path.join(d2, f"f{i:02d}.png"), screen(alert=(i >= 6)))
v = run_gate("seq", "--dir", d2)
print(f"frames in: {v['frames_in']}  passed: {v['keyframes']}  "
      f"dropped: {v['frames_dropped']}")
assert v["keyframes"] == [0, 6], v["keyframes"]
total_in += v["frames_in"]; total_passed += v["frames_passed_to_model"]

pair = run_gate("diff", os.path.join(d2, "f05.png"), os.path.join(d2, "f06.png"))
print(f"change localized: {pair['changed_fraction'] * 100:.2f}% of pixels, "
      f"top region {pair['regions'][0]}")
total_in += pair["frames_in"]; total_passed += pair["frames_passed_to_model"]

try:
    endpoint_up = gate.endpoint_health().get("status") == "ok"
except Exception:
    endpoint_up = False
if endpoint_up:
    pair = run_gate("diff", os.path.join(d2, "f05.png"),
                    os.path.join(d2, "f06.png"),
                    "--ask", "Which region contains the alert dialog that "
                             "replaced the Submit button?")
    dec = pair.get("decision") or {}
    print(f"decision endpoint picked: {dec.get('choice')} (confidence "
          f"{dec.get('confidence')}) decision_id={dec.get('decision_id')}")
else:
    print("(no decision endpoint at $VG_DECISION_URL -- the optional --ask "
          "hook is skipped; the gate's verdict above stands on its own)")

print("\n== Scenario 3: synthetic video keyframe triage (every 3rd frame) ==")
vid = os.path.join(tmp, "synth.avi")
wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
for i in range(90):
    fr = np.full((240, 320, 3), 40, np.uint8)
    if i >= 30:
        x = 10 + (i - 30) * 4
        cv2.rectangle(fr, (x, 90), (x + 60, 150), (255, 255, 255), -1)
    wr.write(fr)
wr.release()
v = run_gate("video", vid, "--every", "3")
print(f"frames in: {v['frames_in']}  keyframes passed: "
      f"{v['frames_passed_to_model']}  dropped: {v['frames_dropped']}")
print(f"keyframes: {v['keyframes']}")
total_in += v["frames_in"]; total_passed += v["frames_passed_to_model"]

print(f"\n== TOTAL across demo: {total_in} frames in, {total_passed} reached "
      f"a model, {total_in - total_passed} kept out "
      f"({100.0 * (total_in - total_passed) / total_in:.1f}%) ==")
