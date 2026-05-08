#!/usr/bin/env python3
"""Quick smoke test — 3 frames, scale=2, CPU fallback."""
import os, sys, time

os.environ["MPLCONFIGDIR"] = "/tmp/comfyui-videoupscalehandfix-mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torchvision.transforms.functional as functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes.video_upscale_hand_fix import (
    _create_upsampler, _detect_hand_boxes, _get_hands_detector,
    _enhance_hand_crop, _feather_blend, _scale_box,
)

out_dir = "/tmp/upscale_test"
os.makedirs(out_dir, exist_ok=True)

# Load 3 frames
VIDEO_PATH = os.environ.get("TEST_VIDEO_PATH")
if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
    print("Set TEST_VIDEO_PATH=/path/to/video.mp4 to run this test.")
    sys.exit(0)
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
for _ in range(3):
    ret, f = cap.read()
    if ret:
        frames.append(f)
cap.release()
print(f"Loaded {len(frames)} frames, shape={frames[0].shape}")

# Use CPU to avoid MPS hang
device = torch.device("cpu")
print(f"Device: {device}")

# ── Test 1: Upscale scale=2 ──
print("\n=== Test 1: Upscale 2x ===")
t0 = time.time()
upsampler = _create_upsampler(2, 256, device)
upscaled_frames = []
for i, frame in enumerate(frames):
    t1 = time.time()
    out, _ = upsampler.enhance(frame, outscale=2.0)
    upscaled_frames.append(out)
    print(f"  Frame {i}: {frame.shape} -> {out.shape} ({time.time()-t1:.1f}s)")
dt = time.time() - t0
print(f"Total upscale: {dt:.1f}s ({dt/len(frames):.2f}s/frame)")

# ── Test 2: Hand detection ──
print("\n=== Test 2: Hand detection ===")
hand_count = 0
for i, frame in enumerate(upscaled_frames):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t1 = time.time()
    hands = _detect_hand_boxes(rgb)
    dt_det = time.time() - t1
    if hands:
        hand_count += len(hands)
        print(f"  Frame {i}: {len(hands)} hands ({dt_det:.3f}s) — {hands}")
    else:
        print(f"  Frame {i}: no hands ({dt_det:.3f}s)")
print(f"Total hands found: {hand_count}")

# ── Test 3: Hand enhancement ──
print("\n=== Test 3: Hand enhancement ===")
if hand_count > 0:
    # Take first frame with hands
    for i, frame in enumerate(upscaled_frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = _detect_hand_boxes(rgb)
        if hands:
            box = hands[0]
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            t1 = time.time()
            enhanced = _enhance_hand_crop(crop, 0.5, 0.3)
            dt_enh = time.time() - t1
            print(f"  Enhanced crop {crop.shape} -> {enhanced.shape} ({dt_enh:.3f}s)")

            t1 = time.time()
            result = _feather_blend(frame.copy(), enhanced, box)
            dt_blend = time.time() - t1
            print(f"  Feather blend ({dt_blend:.3f}s)")

            # Save comparison
            cv2.imwrite(f"{out_dir}/frame{i}_upscaled.png", frame)
            cv2.imwrite(f"{out_dir}/frame{i}_crop_before.png", crop)
            cv2.imwrite(f"{out_dir}/frame{i}_crop_after.png", enhanced)
            cv2.imwrite(f"{out_dir}/frame{i}_blended.png", result)
            print(f"  Saved to {out_dir}/")
            break
else:
    print("  No hands detected — skipping enhancement test")
    # Save upscaled frames anyway
    for i, f in enumerate(upscaled_frames):
        cv2.imwrite(f"{out_dir}/frame{i}_upscaled.png", f)

# ── Test 4: scale=1 (passthrough upscale) ──
print("\n=== Test 4: scale=1 ===")
t0 = time.time()
out1, _ = upsampler.enhance(frames[0], outscale=1.0)
dt = time.time() - t0
print(f"  {frames[0].shape} -> {out1.shape} ({dt:.1f}s)")

print("\n=== ALL TESTS PASSED ===")
