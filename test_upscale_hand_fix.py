#!/usr/bin/env python3
"""
End-to-end smoke test for VideoUpscaleHandFix pipeline.
Tests: ESRGAN upscale, MediaPipe hand detection, hand enhancement.
"""
import os
import sys
import time

os.environ["MPLCONFIGDIR"] = "/tmp/comfyui-videoupscalehandfix-mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torchvision.transforms.functional as functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes.video_upscale_hand_fix import (
    _create_upsampler,
    _preferred_device,
    _detect_hand_boxes,
    _get_hands_detector,
    _enhance_hand_crop,
    _feather_blend,
)

# ─── UTILS ──────────────────────────────────────────────────────────

def load_video_frames(video_path: str, max_frames: int = 16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def upscale_frames(frames, scale=2, tile=256):
    device = _preferred_device()
    upsampler = _create_upsampler(scale, tile, device)
    results = []
    for frame in frames:
        out, _ = upsampler.enhance(frame, outscale=float(scale))
        results.append(out)
    return results


def detect_hands(rgb_frame):
    detector = _get_hands_detector()
    from mediapipe import Image as MpImage, ImageFormat
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return []
    h, w = rgb_frame.shape[:2]
    boxes = []
    for hand in result.hand_landmarks:
        xs = [lm.x for lm in hand]
        ys = [lm.y for lm in hand]
        x1 = int(max(0.0, min(xs)) * w)
        y1 = int(max(0.0, min(ys)) * h)
        x2 = int(min(1.0, max(xs)) * w)
        y2 = int(min(1.0, max(ys)) * h)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad = int(max(box_w, box_h) * 0.35) + 8
        boxes.append((
            max(0, x1 - pad), max(0, y1 - pad),
            min(w, x2 + pad), min(h, y2 + pad),
        ))
    return boxes


def upscale_and_fix_hands(frames, scale=2, tile=256, boost=0.5, denoise=0.3):
    device = _preferred_device()
    upsampler = _create_upsampler(scale, tile, device)
    results = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        upscaled_bgr, _ = upsampler.enhance(frame, outscale=float(scale))

        hands = detect_hands(rgb)
        for box in hands:
            # Scale box from original to upscaled
            oh, ow = frame.shape[:2]
            uh, uw = upscaled_bgr.shape[:2]
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
            crop = upscaled_bgr[y1:y2, x1:x2]
            enhanced = _enhance_hand_crop(crop, boost, denoise)
            upscaled_bgr = _feather_blend(upscaled_bgr, enhanced, scaled)

        results.append(upscaled_bgr)
    return results


# ─── MAIN ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    video_path = os.environ.get("TEST_VIDEO_PATH")
    if not video_path or not os.path.exists(video_path):
        print("Set TEST_VIDEO_PATH=/path/to/video.mp4 to run this test.")
        sys.exit(0)
    out_dir = os.environ.get("TEST_OUTPUT_DIR", "/tmp/upscale_test")
    os.makedirs(out_dir, exist_ok=True)

    frames = load_video_frames(video_path, max_frames=20)
    print(f"Loaded {len(frames)} frames, shape={frames[0].shape}")

    # ── Test 1: Full upscale (scale=2) ──
    t0 = time.time()
    up2 = upscale_frames(frames, scale=2, tile=256)
    dt_up2 = time.time() - t0
    h2, w2 = up2[0].shape[:2]
    print(f"Upscale 2x: {len(up2)} frames {w2}x{h2} in {dt_up2:.1f}s ({dt_up2/len(up2):.2f}s/frame)")

    # ── Test 2: Full upscale (scale=4) ──
    t0 = time.time()
    up4 = upscale_frames(frames, scale=4, tile=256)
    dt_up4 = time.time() - t0
    h4, w4 = up4[0].shape[:2]
    print(f"Upscale 4x: {len(up4)} frames {w4}x{h4} in {dt_up4:.1f}s ({dt_up4/len(up4):.2f}s/frame)")

    # ── Test 3: Hand detection on upscaled frames ──
    t0 = time.time()
    hand_count = 0
    for i, frame in enumerate(up2):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detect_hands(rgb)
        if hands:
            hand_count += len(hands)
            print(f"  Frame {i}: {len(hands)} hands at {[(h[0],h[1],h[2],h[3]) for h in hands]}")
    dt_hand = time.time() - t0
    print(f"Hand detection: {hand_count} hands found in {len(up2)} frames ({dt_hand:.1f}s)")

    # ── Test 4: Full pipeline (upscale + hand fix) on scale=2 ──
    t0 = time.time()
    fixed = upscale_and_fix_hands(frames, scale=2, tile=256, boost=0.5, denoise=0.3)
    dt_full = time.time() - t0
    print(f"Full pipeline (2x): {len(fixed)} frames in {dt_full:.1f}s ({dt_full/len(fixed):.2f}s/frame)")

    # ── Save outputs ──
    cv2.imwrite(f"{out_dir}/orig_f0.png", frames[0])
    cv2.imwrite(f"{out_dir}/upscaled2x_f0.png", up2[0])
    cv2.imwrite(f"{out_dir}/upscaled4x_f0.png", up4[0])
    cv2.imwrite(f"{out_dir}/handfix2x_f0.png", fixed[0])

    # Side-by-side comparison for hand region
    # Use last frame (typically more hands visible in dance)
    last = len(frames) - 1
    cv2.imwrite(f"{out_dir}/orig_f{last}.png", frames[last])
    cv2.imwrite(f"{out_dir}/upscaled2x_f{last}.png", up2[last])
    cv2.imwrite(f"{out_dir}/handfix2x_f{last}.png", fixed[last])

    print(f"\nResults saved to {out_dir}/")
    print("=== TEST PASSED ===" if len(up2) == len(frames) else "=== TEST FAILED ===")
