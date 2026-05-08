import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch


def _comfyui_base() -> Path:
    """Resolve ComfyUI base directory dynamically."""
    try:
        import folder_paths
        return Path(folder_paths.models_dir).parent
    except Exception:
        # nodes/video_upscale_hand_fix.py -> nodes/ -> ComfyUI-VideoUpscaleHandFix/ -> custom_nodes/ -> ComfyUI/
        return Path(__file__).resolve().parent.parent.parent.parent

MODEL_DIR_FALLBACK = _comfyui_base() / "models" / "upscale_models"

# arch="rrdb"  -> basicsr.archs.rrdbnet_arch.RRDBNet (heavy, high quality)
# arch="srvgg" -> realesrgan.archs.srvgg_arch.SRVGGNetCompact (small, video-tuned)
MODEL_SPECS = {
    "RealESRGAN_x2plus": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "arch": "rrdb",
        "num_block": 23,
    },
    "RealESRGAN_x4plus": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "arch": "rrdb",
        "num_block": 23,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "arch": "rrdb",
        "num_block": 6,
    },
    "realesr-animevideov3": {
        "filename": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "arch": "srvgg",
        "num_conv": 16,
    },
    "realesr-general-x4v3": {
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale": 4,
        "arch": "srvgg",
        "num_conv": 32,
    },
}
DEFAULT_MODEL = "RealESRGAN_x2plus"

_UPSAMPLER_CACHE = {}
_HANDS = None


def _download_with_progress(url: str, dest: Path, label: str, timeout: float = 30.0) -> None:
    """Download URL to dest with a connection timeout and 10%-step progress log."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "ComfyUI-VideoUpscaleHandFix"}
    )
    print(f"[VideoUpscaleHandFix] Downloading {label} from {url}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunk_size = 1 << 16
        downloaded = 0
        last_pct = -10
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct - last_pct >= 10:
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(
                            f"[VideoUpscaleHandFix] {label}: {pct}% "
                            f"({mb:.1f}/{total_mb:.1f} MB)"
                        )
                        last_pct = pct
        mb = downloaded / (1024 * 1024)
        print(f"[VideoUpscaleHandFix] {label}: done ({mb:.1f} MB)")


def _model_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "upscale_models"
    except Exception:
        return MODEL_DIR_FALLBACK


def _ensure_torchvision_compat():
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except Exception:
        pass


def _require_cv2():
    try:
        import cv2

        return cv2
    except Exception as exc:
        raise RuntimeError(
            "VideoUpscaleHandFix requires opencv-python. Install it in the ComfyUI environment."
        ) from exc


def _require_realesrgan():
    _ensure_torchvision_compat()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.utils import RealESRGANer

        return RRDBNet, RealESRGANer
    except Exception as exc:
        raise RuntimeError(
            "VideoUpscaleHandFix requires compatible realesrgan/basicsr packages. "
            "Install realesrgan, basicsr, gfpgan, and facexlib in the ComfyUI environment."
        ) from exc


HAND_MODEL_DIR = _comfyui_base() / "models" / "mediapipe"
HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_hand_model() -> str:
    """Download hand_landmarker.task if missing."""
    HAND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = HAND_MODEL_DIR / HAND_MODEL_NAME
    if model_path.exists() and model_path.stat().st_size > 0:
        return str(model_path)

    temp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        _download_with_progress(HAND_MODEL_URL, temp_path, HAND_MODEL_NAME)
        temp_path.replace(model_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return str(model_path)


def _require_mediapipe():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "comfyui-videoupscalehandfix-mpl"),
    )
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        return BaseOptions, HandLandmarker, HandLandmarkerOptions, RunningMode
    except Exception as exc:
        raise RuntimeError(
            "VideoUpscaleHandFix requires mediapipe >= 0.10 with Tasks API (HandLandmarker)."
        ) from exc


def _download_model(model_name: str) -> Path:
    spec = MODEL_SPECS[model_name]
    model_dir = _model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / spec["filename"]
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    temp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        _download_with_progress(spec["url"], temp_path, spec["filename"])
        temp_path.replace(model_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return model_path


def _preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(spec: dict):
    arch = spec["arch"]
    if arch == "rrdb":
        RRDBNet, _ = _require_realesrgan()
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=spec["num_block"],
            num_grow_ch=32,
            scale=spec["scale"],
        )
    if arch == "srvgg":
        _ensure_torchvision_compat()
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=spec["num_conv"],
            upscale=spec["scale"],
            act_type="prelu",
        )
    raise ValueError(f"Unknown model arch: {arch!r}")


def _create_upsampler(model_name: str, tile_size: int, device: torch.device):
    _, RealESRGANer = _require_realesrgan()
    spec = MODEL_SPECS[model_name]
    model_path = _download_model(model_name)
    model = _build_model(spec)
    return RealESRGANer(
        scale=spec["scale"],
        model_path=str(model_path),
        model=model,
        tile=max(0, int(tile_size)),
        tile_pad=10,
        pre_pad=10,
        half=device.type == "cuda",
        device=device,
    )


def _get_upsampler(model_name: str, tile_size: int, device: torch.device):
    key = (model_name, int(tile_size), device.type)
    if key not in _UPSAMPLER_CACHE:
        try:
            _UPSAMPLER_CACHE[key] = _create_upsampler(model_name, tile_size, device)
        except Exception:
            if device.type == "mps":
                cpu_device = torch.device("cpu")
                key = (model_name, int(tile_size), cpu_device.type)
                if key not in _UPSAMPLER_CACHE:
                    print("[VideoUpscaleHandFix] MPS init failed; falling back to CPU.")
                    _UPSAMPLER_CACHE[key] = _create_upsampler(
                        model_name, tile_size, cpu_device
                    )
            else:
                raise
    return _UPSAMPLER_CACHE[key]


def _run_realesrgan(
    frame_bgr: np.ndarray, model_name: str, scale_factor: int, tile_size: int
):
    device = _preferred_device()
    upsampler = _get_upsampler(model_name, tile_size, device)

    try:
        output, _ = upsampler.enhance(frame_bgr, outscale=float(scale_factor))
        return output
    except Exception:
        if device.type != "mps":
            raise

        print("[VideoUpscaleHandFix] MPS inference failed; retrying this frame on CPU.")
        cpu_upsampler = _get_upsampler(model_name, tile_size, torch.device("cpu"))
        output, _ = cpu_upsampler.enhance(frame_bgr, outscale=float(scale_factor))
        return output


def _get_hands_detector():
    global _HANDS
    if _HANDS is None:
        BaseOptions, HandLandmarker, HandLandmarkerOptions, RunningMode = _require_mediapipe()
        model_path = _ensure_hand_model()
        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        _HANDS = HandLandmarker.create_from_options(options)
    return _HANDS


def _detect_hand_boxes(rgb_frame: np.ndarray, padding_ratio: float = 0.35):
    """Detect hand bounding boxes using MediaPipe HandLandmarker (Tasks API)."""
    from mediapipe import Image as MpImage, ImageFormat

    detector = _get_hands_detector()
    height, width = rgb_frame.shape[:2]

    # MediaPipe Image expects RGB uint8
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb_frame)

    try:
        result = detector.detect(mp_image)
    except Exception as exc:
        print(f"[VideoUpscaleHandFix] Hand detection failed: {exc!r}")
        return []

    if not result.hand_landmarks:
        return []

    padding_ratio = max(0.0, float(padding_ratio))
    boxes = []
    for hand_landmarks in result.hand_landmarks:
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]
        x1 = int(max(0.0, min(xs)) * width)
        y1 = int(max(0.0, min(ys)) * height)
        x2 = int(min(1.0, max(xs)) * width)
        y2 = int(min(1.0, max(ys)) * height)

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        padding = int(max(box_w, box_h) * padding_ratio) + 8
        boxes.append(
            (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(width, x2 + padding),
                min(height, y2 + padding),
            )
        )
    return boxes


def _scale_box(box, source_shape, target_shape):
    src_h, src_w = source_shape[:2]
    dst_h, dst_w = target_shape[:2]
    x1, y1, x2, y2 = box
    sx = dst_w / src_w
    sy = dst_h / src_h
    return (
        max(0, min(dst_w, int(round(x1 * sx)))),
        max(0, min(dst_h, int(round(y1 * sy)))),
        max(0, min(dst_w, int(round(x2 * sx)))),
        max(0, min(dst_h, int(round(y2 * sy)))),
    )


def _enhance_hand_crop(crop_bgr: np.ndarray, strength: float, denoise_strength: float):
    cv2 = _require_cv2()
    height, width = crop_bgr.shape[:2]
    if height < 2 or width < 2:
        return crop_bgr

    boosted = cv2.resize(
        crop_bgr,
        (width * 2, height * 2),
        interpolation=cv2.INTER_LANCZOS4,
    )
    boosted = cv2.resize(boosted, (width, height), interpolation=cv2.INTER_LANCZOS4)

    strength = float(strength)
    if strength > 0.0:
        amount = min(1.5, 0.75 * strength)
        sigma = 1.0 + min(2.0, strength) * 0.4
        blurred = cv2.GaussianBlur(boosted, (0, 0), sigmaX=sigma, sigmaY=sigma)
        boosted = cv2.addWeighted(boosted, 1.0 + amount, blurred, -amount, 0)

    denoise_strength = float(denoise_strength)
    if denoise_strength > 0.0:
        sigma_color = 25.0 + denoise_strength * 50.0
        sigma_space = 7.0 + denoise_strength * 8.0
        boosted = cv2.bilateralFilter(
            boosted,
            d=0,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )

    return boosted


def _feather_blend(
    base_bgr: np.ndarray, enhanced_bgr: np.ndarray, box, feather_ratio: float = 0.12
):
    cv2 = _require_cv2()
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return base_bgr

    crop = base_bgr[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    feather_ratio = max(0.0, float(feather_ratio))
    feather = max(3, int(min(width, height) * feather_ratio))
    if feather % 2 == 0:
        feather += 1

    padded = np.zeros((height + feather * 2, width + feather * 2), dtype=np.float32)
    padded[feather : feather + height, feather : feather + width] = 1.0
    mask = cv2.GaussianBlur(padded, (feather, feather), 0)
    mask = mask[feather : feather + height, feather : feather + width]
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]

    blended = enhanced_bgr.astype(np.float32) * mask + crop.astype(np.float32) * (1.0 - mask)
    base_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return base_bgr


def _normalize_images_tensor(images):
    if not torch.is_tensor(images):
        raise TypeError("images must be a ComfyUI IMAGE tensor.")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (N, H, W, 3).")
    return images.detach().cpu().float().clamp(0.0, 1.0)


class VideoUpscaleHandFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                    },
                ),
                "hand_enhance": ("BOOLEAN", {"default": True}),
                "hand_boost_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "denoise_strength": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 0,
                        "max": 512,
                        "step": 64,
                    },
                ),
                "model": (
                    list(MODEL_SPECS.keys()),
                    {"default": DEFAULT_MODEL},
                ),
                "hand_padding_ratio": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "feather_ratio": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                    },
                ),
            }
        }

    @classmethod
    def GET_SCHEMA(cls):
        return cls.INPUT_TYPES()

    CATEGORY = "video"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = "Video frame Real-ESRGAN upscale with MediaPipe hand crop enhancement."

    def upscale(
        self,
        images,
        scale_factor=2,
        hand_enhance=True,
        hand_boost_strength=0.5,
        denoise_strength=0.3,
        tile_size=256,
        model=DEFAULT_MODEL,
        hand_padding_ratio=0.35,
        feather_ratio=0.12,
    ):
        cv2 = _require_cv2()
        images = _normalize_images_tensor(images)
        output_frames = []
        scale_factor = int(scale_factor)
        skip_esrgan = scale_factor <= 1
        if model not in MODEL_SPECS:
            print(
                f"[VideoUpscaleHandFix] Unknown model {model!r}, "
                f"falling back to {DEFAULT_MODEL!r}"
            )
            model = DEFAULT_MODEL

        for index, frame in enumerate(images):
            rgb = (frame.numpy() * 255.0).round().astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if skip_esrgan:
                print(
                    f"[VideoUpscaleHandFix] Frame {index + 1}/{len(images)} "
                    f"(passthrough, scale=1)"
                )
                upscaled_bgr = bgr
            else:
                print(
                    f"[VideoUpscaleHandFix] Upscaling frame "
                    f"{index + 1}/{len(images)} with {model}"
                )
                upscaled_bgr = _run_realesrgan(
                    bgr, model, scale_factor, int(tile_size)
                )

            if hand_enhance:
                boxes = _detect_hand_boxes(rgb, padding_ratio=float(hand_padding_ratio))
                for box in boxes:
                    scaled_box = _scale_box(box, rgb.shape, upscaled_bgr.shape)
                    x1, y1, x2, y2 = scaled_box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = upscaled_bgr[y1:y2, x1:x2]
                    enhanced = _enhance_hand_crop(
                        crop,
                        float(hand_boost_strength),
                        float(denoise_strength),
                    )
                    upscaled_bgr = _feather_blend(
                        upscaled_bgr,
                        enhanced,
                        scaled_box,
                        feather_ratio=float(feather_ratio),
                    )

            upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
            output_frames.append(upscaled_rgb.astype(np.float32) / 255.0)

        output = torch.from_numpy(np.stack(output_frames, axis=0)).clamp(0.0, 1.0)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "VideoUpscaleHandFix": VideoUpscaleHandFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUpscaleHandFix": "Video Upscale Hand Fix",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "VideoUpscaleHandFix"]
