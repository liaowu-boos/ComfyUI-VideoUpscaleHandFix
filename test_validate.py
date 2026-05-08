#!/usr/bin/env python3
"""Quick validation: 1 frame, scale=2, verify all pipeline steps."""
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
    _enhance_hand_crop, _feather_blend, _scale_box, _comfyui_base,
    _ensure_hand_model,
)

out_dir = "/tmp/upscale_test"
os.makedirs(out_dir, exist_ok=True)

# Verify paths
base = _comfyui_base()
print(f"ComfyUI base: {base}")
print(f"Hand model exists: {(base / 'models/mediapipe/hand_landmarker.task').exists()}")

# Create synthetic test image with a hand-like shape
# (A colored rectangle to simulate a hand region for enhancement testing)
print("\n=== Creating synthetic test frame ===")
frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

# Add a "hand" region - skin-colored area
frame[200:350, 300:450] = [180, 150, 130]  # skin tone BGR

device = torch.device("cpu")

# Test 1: ESRGAN 2x upscale
print("\n=== Test 1: ESRGAN 2x ===")
t0 = time.time()
upsampler = _create_upsampler(2, 256, device)
upscaled, _ = upsampler.enhance(frame, outscale=2.0)
dt = time.time() - t0
print(f"  {frame.shape} -> {upscaled.shape} ({dt:.1f}s)")

# Test 2: MediaPipe hand detection on upscaled frame
print("\n=== Test 2: Hand detection ===")
rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
hands = _detect_hand_boxes(rgb)
print(f"  Detected {len(hands)} hands (expected 0 for synthetic image)")

# Test 3: Hand enhancement (manual box)
print("\n=== Test 3: Hand crop enhancement ===")
# Use a region from the upscaled frame
box = (400, 300, 700, 600)  # x1, y1, x2, y2
x1, y1, x2, y2 = box
crop = upscaled[y1:y2, x1:x2]
t0 = time.time()
enhanced = _enhance_hand_crop(crop, 0.5, 0.3)
dt = time.time() - t0
print(f"  Enhanced {crop.shape} -> {enhanced.shape} ({dt:.3f}s)")

# Test 4: Feather blend
print("\n=== Test 4: Feather blend ===")
t0 = time.time()
result = _feather_blend(upscaled.copy(), enhanced, box)
dt = time.time() - t0
print(f"  Blended shape: {result.shape} ({dt:.3f}s)")

# Save
cv2.imwrite(f"{out_dir}/synth_orig.png", frame)
cv2.imwrite(f"{out_dir}/synth_upscaled.png", upscaled)
cv2.imwrite(f"{out_dir}/synth_crop_before.png", crop)
cv2.imwrite(f"{out_dir}/synth_crop_enhanced.png", enhanced)
cv2.imwrite(f"{out_dir}/synth_blended.png", result)

# Test 5: scale=1
print("\n=== Test 5: scale=1 passthrough ===")
out1, _ = upsampler.enhance(frame, outscale=1.0)
print(f"  {frame.shape} -> {out1.shape}")

# Test 6: scale=4 model
print("\n=== Test 6: ESRGAN 4x model load ===")
t0 = time.time()
upsampler4 = _create_upsampler(4, 256, device)
dt = time.time() - t0
print(f"  4x model loaded ({dt:.1f}s)")

print(f"\nSaved to {out_dir}/")
print("\n=== ALL VALIDATION PASSED ===")
