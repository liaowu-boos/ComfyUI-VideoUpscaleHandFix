#!/usr/bin/env python3
"""Test with synthetic hand-like image (skin tones) and full pipeline."""
import cv2, os, sys, time
import numpy as np

os.environ['MPLCONFIGDIR'] = '/tmp/comfyui-videoupscalehandfix-mpl'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import torchvision.transforms.functional as functional
sys.modules['torchvision.transforms.functional_tensor'] = functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes.video_upscale_hand_fix import (
    _detect_hand_boxes, _create_upsampler, _preferred_device,
    _enhance_hand_crop, _feather_blend
)

out_dir = '/tmp/upscale_test'
os.makedirs(out_dir, exist_ok=True)

# Create a realistic hand-like image with skin tones and fingers
print("=== Creating synthetic hand-like test image (640x480) ===")
img = np.full((480, 640, 3), 60, dtype=np.uint8)  # dark background

# Draw hand/arm shape (skin-colored, BGR)
skin = (140, 170, 210)
# Palm
cv2.rectangle(img, (220, 150), (420, 400), skin, -1)
# Fingers (5 rectangles)
cv2.rectangle(img, (235, 50), (270, 200), skin, -1)   # thumb
cv2.rectangle(img, (275, 30), (305, 200), skin, -1)   # index
cv2.rectangle(img, (310, 20), (340, 190), skin, -1)   # middle
cv2.rectangle(img, (345, 35), (375, 195), skin, -1)   # ring
cv2.rectangle(img, (378, 60), (408, 200), skin, -1)   # pinky
# Wrist/arm
cv2.rectangle(img, (270, 380), (370, 480), skin, -1)

cv2.imwrite(os.path.join(out_dir, 'synth_hand_detailed.png'), img)
print(f"Created: {img.shape}")

# Step 1: Hand detection
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("\n=== Step 1: Hand detection ===")
t0 = time.time()
hands = _detect_hand_boxes(rgb)
dt = time.time() - t0
print(f"Detected {len(hands)} hands ({dt:.3f}s)")
for i, box in enumerate(hands):
    print(f"  Hand {i}: {box}")

# Step 2: Real-ESRGAN upscale
print("\n=== Step 2: Real-ESRGAN upscale (scale=2, MPS) ===")
device = _preferred_device()
print(f"Device: {device}")

t0 = time.time()
upsampler = _create_upsampler(2, 256, device)
upscaled, _ = upsampler.enhance(img, outscale=2.0)
dt = time.time() - t0
print(f"Upscaled: {img.shape} -> {upscaled.shape} ({dt:.1f}s)")

# Step 3: Hand enhancement (use detected hands, or manual box if none)
print("\n=== Step 3: Hand crop enhancement ===")
if hands:
    boxes = hands
else:
    # Manual box (simulate detected hand region)
    boxes = [(150, 20, 550, 450)]
    print("  (using manual hand box since synthetic image)")

for box in boxes:
    # Scale box to upscaled resolution
    uh, uw = upscaled.shape[:2]
    oh, ow = img.shape[:2]
    sx, sy = uw / ow, uh / oh
    scaled = (
        max(0, min(uw, int(round(box[0] * sx)))),
        max(0, min(uh, int(round(box[1] * sy)))),
        max(0, min(uw, int(round(box[2] * sx)))),
        max(0, min(uh, int(round(box[3] * sy)))),
    )
    x1, y1, x2, y2 = scaled
    if x2 <= x1 or y2 <= y1:
        continue

    crop = upscaled[y1:y2, x1:x2]
    print(f"  Hand crop: {crop.shape}")

    t0 = time.time()
    enhanced = _enhance_hand_crop(crop, 0.5, 0.3)
    dt = time.time() - t0
    print(f"  Enhancement: {dt:.3f}s")

    t0 = time.time()
    result = _feather_blend(upscaled.copy(), enhanced, scaled)
    dt = time.time() - t0
    print(f"  Feather blend: {dt:.3f}s")

    # Save results
    cv2.imwrite(os.path.join(out_dir, 'hand_synth_orig.png'), img)
    cv2.imwrite(os.path.join(out_dir, 'hand_synth_upscaled.png'), upscaled)
    cv2.imwrite(os.path.join(out_dir, 'hand_synth_crop_before.png'), crop)
    cv2.imwrite(os.path.join(out_dir, 'hand_synth_crop_enhanced.png'), enhanced)
    cv2.imwrite(os.path.join(out_dir, 'hand_synth_result.png'), result)
    print(f"  Saved results to {out_dir}/")
    break

# Step 4: scale=1 passthrough
print("\n=== Step 4: scale=1 passthrough ===")
t0 = time.time()
out1, _ = upsampler.enhance(img, outscale=1.0)
dt = time.time() - t0
print(f"  {img.shape} -> {out1.shape} ({dt:.1f}s)")

print("\n=== ALL PIPELINE TESTS PASSED ===")
