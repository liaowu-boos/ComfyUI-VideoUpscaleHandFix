#!/usr/bin/env python3
"""MPS end-to-end test: 3 frames, scale=2, with hand detection."""
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
    _create_upsampler, _preferred_device, _detect_hand_boxes,
    _get_hands_detector, _enhance_hand_crop, _feather_blend, _scale_box,
)

out_dir = "/tmp/upscale_test"
os.makedirs(out_dir, exist_ok=True)

# Load 3 frames from video
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

device = _preferred_device()
print(f"Device: {device}")

# Test 1: ESRGAN 2x upscale (MPS)
print("\n=== Test 1: ESRGAN 2x (MPS) ===")
upsampler = _create_upsampler(2, 256, device)
upscaled_frames = []
for i, frame in enumerate(frames):
    t0 = time.time()
    out, _ = upsampler.enhance(frame, outscale=2.0)
    upscaled_frames.append(out)
    dt = time.time() - t0
    print(f"  Frame {i}: {frame.shape} -> {out.shape} ({dt:.1f}s)")

# Test 2: Hand detection
print("\n=== Test 2: Hand detection ===")
total_hands = 0
for i, frame in enumerate(upscaled_frames):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = _detect_hand_boxes(rgb)
    if hands:
        total_hands += len(hands)
        for j, box in enumerate(hands):
            print(f"  Frame {i}: Hand {j} at {box}")
    else:
        print(f"  Frame {i}: no hands")

# Test 3: Hand enhancement (if any hands found)
print(f"\n=== Test 3: Hand enhancement ({total_hands} hands total) ===")
if total_hands > 0:
    for i, frame in enumerate(upscaled_frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = _detect_hand_boxes(rgb)
        for box in hands:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            t0 = time.time()
            enhanced = _enhance_hand_crop(crop, 0.5, 0.3)
            dt = time.time() - t0
            print(f"  Enhanced {crop.shape} ({dt:.3f}s)")
            result = _feather_blend(frame.copy(), enhanced, box)
            cv2.imwrite(f"{out_dir}/frame{i}_hand_enhanced.png", result)
            cv2.imwrite(f"{out_dir}/frame{i}_upscaled.png", frame)
            print(f"  Saved to {out_dir}/")
            break
        else:
            continue
        break
else:
    print("  No hands found — saving upscaled frame as proof")
    cv2.imwrite(f"{out_dir}/frame0_upscaled.png", upscaled_frames[0])

print("\n=== MPS END-TO-END TEST PASSED ===")
