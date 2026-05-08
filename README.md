# ComfyUI-VideoUpscaleHandFix

一个 ComfyUI 自定义节点，针对**真人舞蹈视频手部不清晰**的问题：先用 Real-ESRGAN 对每帧整体放大，再用 MediaPipe 检测手部、对手部区域做额外锐化与去噪，最后羽化融合回去。

确定性算法 + 单帧推理，不引入生成模型，**保证视频时序连贯**，不会出现帧间手指数量/朝向跳变的问题。

## 功能

- 5 个可选 SR 模型（含视频专用的轻量 SRVGGNetCompact，0.6M / 1.2M 参数）
- MediaPipe HandLandmarker 检测手部 bbox
- 手部区域 unsharp mask 锐化 + bilateralFilter 去噪
- 高斯羽化 alpha 融合，避免接缝
- 模型自动下载（带超时和进度日志）
- MPS / CUDA / CPU 自动选择，MPS 推理失败自动 fallback CPU
- 模型与 detector 模块级缓存，多帧复用

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liaowu-boos/ComfyUI-VideoUpscaleHandFix.git
cd ComfyUI-VideoUpscaleHandFix
pip install -r requirements.txt
```

依赖：`opencv-python`、`mediapipe>=0.10`、`realesrgan`、`basicsr`、`gfpgan`、`facexlib`、`scipy`。

> **torchvision 兼容性**：节点会自动给 `torchvision.transforms.functional_tensor` 打 shim，无需手动处理新版 torchvision 移除该模块的问题。

首次运行会自动下载：
- ESRGAN 模型 → `ComfyUI/models/upscale_models/`
- MediaPipe 手部模型 → `ComfyUI/models/mediapipe/`

## 使用

工作流示例见 `workflow_upscale_hand_fix.json`。基本接线：

```
VHS_LoadVideo ──IMAGE──> VideoUpscaleHandFix ──IMAGE──> VHS_VideoCombine
```

节点位于 ComfyUI 的 `video` 分类下。

## 参数说明

### 输入
| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images` | IMAGE | — | 视频帧序列 `(N, H, W, 3)`，float32 [0,1] |

### 模型与放大
| 参数 | 类型 | 默认 | 范围 | 说明 |
|---|---|---|---|---|
| `model` | 下拉 | `RealESRGAN_x2plus` | 见下表 | 选择 SR 模型 |
| `scale_factor` | INT | 2 | 1-4 | 输出放大倍数；`1` 直接 passthrough，跳过模型 |
| `tile_size` | INT | 256 | 0-512（步进 64） | 分块大小；`0` 不分块（最快、最吃显存） |

#### 5 个可选模型

| 名称 | 架构 | 参数量 | 适合 |
|---|---|---|---|
| `RealESRGAN_x2plus` | RRDB-23 | 16.7M | 通用 2x，质量稳定 |
| `RealESRGAN_x4plus` | RRDB-23 | 16.7M | 通用 4x |
| `RealESRGAN_x4plus_anime_6B` | RRDB-6 | 4.5M | 动漫，真人慎用 |
| `realesr-animevideov3` | SRVGG-16 | **0.62M** | 视频专用，速度极快 |
| `realesr-general-x4v3` | SRVGG-32 | **1.21M** | **真人视频首选** |

> 真人舞蹈推荐顺序：`realesr-general-x4v3` > `realesr-animevideov3` > `RealESRGAN_x2plus`

### 手部增强
| 参数 | 类型 | 默认 | 范围 | 说明 |
|---|---|---|---|---|
| `hand_enhance` | BOOL | True | — | 是否增强手部；不露手的视频建议关闭 |
| `hand_boost_strength` | FLOAT | 0.5 | 0.0-2.0 | unsharp mask 锐化强度；超过 1.2 易出光晕 |
| `denoise_strength` | FLOAT | 0.3 | 0.0-1.0 | bilateralFilter 去噪强度；与锐化对抗 |
| `hand_padding_ratio` | FLOAT | 0.35 | 0.0-1.0 | 手部 bbox 外扩比例；切到指尖时调大 |
| `feather_ratio` | FLOAT | 0.12 | 0.0-0.5 | 羽化融合宽度；看到接缝时调大 |

### 输出
| 参数 | 类型 | 说明 |
|---|---|---|
| `images` | IMAGE | 放大后的帧序列 |

## 调参速查

| 你想要 | 配置 |
|---|---|
| **最快预览** | `model=realesr-animevideov3, scale=2, hand_enhance=False, tile=0` |
| **质量优先（真人）** | `model=realesr-general-x4v3, scale=2, hand_enhance=True, boost=0.6, denoise=0.3, tile=256` |
| **手部细节最大化** | `model=realesr-general-x4v3, scale=2, hand_enhance=True, boost=0.8, denoise=0.2, tile=128` |
| **运动模糊场景** | `model=realesr-general-x4v3, scale=2, hand_enhance=True, boost=0.3, denoise=0.5` |

| 现象 | 调整 |
|---|---|
| 手指被裁掉 | `hand_padding_ratio` ↑ 到 0.5+ |
| 增强溢出到衣服/脸 | `hand_padding_ratio` ↓ 到 0.2 |
| 手部边缘有接缝 | `feather_ratio` ↑ 到 0.2 |
| 增强效果太弱 | `feather_ratio` ↓ 到 0.05 |
| 手部"塑料感" | `hand_boost_strength` ↓，`denoise_strength` ↑ |
| 出现暗边光晕 | `hand_boost_strength` ↓ 到 0.5 以下 |

## 设计取舍

**为什么不用生成式模型修手？**

真人舞蹈手部运动幅度大、速度快，diffusion / inpainting 类模型难以保证帧间一致性 — 同一只手在前后帧可能生成不同的指节数量或朝向，整段视频会出现明显抖动。本节点选择**确定性增强**路线（SR 模型 + 算法后处理），输出可重现，时序天然稳定。

**手部检测在原帧而非放大帧上做**：MediaPipe 对小图也能检，省下放大帧的检测成本，bbox 再按比例 scale 到放大帧。

**Real-ESRGAN 模型权重源自 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 官方 release**。

## 技术架构

```
原帧 (H,W,3, RGB) ─┬─> Real-ESRGAN ─> 放大帧 (sH,sW,3, BGR)
                   │
                   └─> MediaPipe ────> 手部 bbox 列表
                                              │
                                              ▼
                            crop -> unsharp -> bilateral
                                              │
                                              ▼
                                    高斯羽化 alpha blend
                                              │
                                              ▼
                                          最终帧
```

## 性能参考（Apple M4 / MPS）

| 配置 | 处理速度（576×320 输入 → 1152×640） |
|---|---|
| `realesr-animevideov3` + `tile=0` + `hand_enhance=False` | ~0.3s/帧 |
| `realesr-general-x4v3` + `tile=256` + `hand_enhance=True` | ~1.5s/帧 |
| `RealESRGAN_x2plus` + `tile=256` + `hand_enhance=True` | ~3s/帧 |

## License

代码使用 MIT。模型权重遵循 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) 与 [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) 各自的 license。
