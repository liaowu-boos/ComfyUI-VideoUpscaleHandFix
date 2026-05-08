# ComfyUI VideoUpscaleHandFix — 视频高清放大（修手）节点

## 目标
创建一个 ComfyUI 自定义节点，针对舞蹈模板视频手部不清晰的问题：
1. 对视频每帧进行高清放大（Real-ESRGAN）
2. 检测手部区域，对手部进行额外增强（锐化 + 二次放大）
3. 输出放大后的视频帧序列（IMAGE tensor）

## 目录结构
```
ComfyUI-VideoUpscaleHandFix/
├── __init__.py              # NODE_CLASS_MAPPINGS 注册
├── nodes/
│   ├── __init__.py
│   └── video_upscale_hand_fix.py   # 核心节点
├── README.md
└── TASK.md
```

## 节点设计

### 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| images | IMAGE | 视频帧序列，shape (N, H, W, 3)，float32 [0,1] |
| scale_factor | INT (1-4, default=2) | 放大倍数 |
| hand_enhance | BOOLEAN (default=True) | 是否对手部区域额外增强 |
| hand_boost_strength | FLOAT (0.0-2.0, default=0.5) | 手部增强强度 (0=仅放大，1=标准锐化，2=强锐化) |
| denoise_strength | FLOAT (0.0-1.0, default=0.3) | 去噪强度 |
| tile_size | INT (64-512, default=256) | Real-ESRGAN tile 大小（省内存） |

### 输出
| 名称 | 类型 | 说明 |
|------|------|------|
| images | IMAGE | 放大后的帧序列 |

### 技术方案

#### 1. 全帧 Real-ESRGAN 放大
- 使用 `realesrgan.archs.srvgg_arch` 的 RealESRGANer
- 模型: RealESRGAN_x2plus（2x）或 RealESRGAN_x4plus（4x）
- 模型自动下载到 `<ComfyUI>/models/upscale_models/`
- MPS 兼容：注意 MPS 部分操作不支持，需要 fallback 到 CPU
- tile 模式避免内存溢出

#### 2. 手部检测（MediaPipe Hands）
```python
import mediapipe as mp
hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)
results = hands.process(rgb_frame)
```
- 检测到手后，获取 bounding box（加 padding）
- 检测失败时跳过手部增强

#### 3. 手部区域增强
对检测到的手部区域：
1. crop 手部区域
2. 对 crop 区域进行额外的锐化处理（Unsharp Mask）
3. 对 crop 区域进行轻度 denoise（去噪减少放大伪影）
4. 羽化融合回放大后的全帧

#### 4. 羽化融合
- 创建手部区域的 mask（带高斯模糊边缘）
- 用 mask 将增强后的手部区域 alpha blend 回全帧

## 关键注意事项

### MPS 兼容性
Mac MPS 后端部分算子不支持，需要 try/except fallback 到 CPU：
```python
try:
    model = model.to('mps')
except:
    model = model.to('cpu')
```

### ComfyUI 0.8.2 插件格式
- 必须有 `GET_SCHEMA` classmethod
- 使用 `NODE_CLASS_MAPPINGS`（非 ComfyExtension）
- CATEGORY 用已知类别如 `"image"` 或 `"video"`

### IMAGE tensor 格式
ComfyUI IMAGE tensor: (N, H, W, 3), float32, range [0, 1]
Real-ESRGAN 使用 numpy BGR uint8 格式，需要转换：
- torch [0,1] → numpy [0,255] uint8
- RGB → BGR
- 输出时反过来

### 模型路径
模型下载到：`<ComfyUI>/models/upscale_models/`
如果该目录为空或模型不存在，节点应自动下载。

### 测试命令
```bash
cd <ComfyUI>  # 替换成你的 ComfyUI 安装路径
./venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from custom_nodes.ComfyUI-VideoUpscaleHandFix.nodes.video_upscale_hand_fix import VideoUpscaleHandFix
# ... test with dummy data
"
```

## 依赖
- realesrgan
- basicsr
- gfpgan
- facexlib
- mediapipe (已安装 0.10.33)
- opencv-python (已安装)
- torch (已安装)
