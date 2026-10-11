# FranchiseFlow

**一条移动门店视频，生成 1 到 10 条分店本地化成片。**

FranchiseFlow 面向连锁品牌的视频本地化制作。用户上传一条连续门店运镜视频和若干分店招牌图片，系统先让视觉模型提出平面候选，再用三帧检查判断候选能否交给本地几何跟踪器。候选不可靠时，页面展示原因并允许拖动四个角点确认。

当前产品原型支持 720p、60 秒以内的视频，自动识别最多 8 个高置信度硬切场景；每个场景最多处理 4 个经过验证且互不重叠的平面，分支数量为 1 到 10 个。多平面任务会记录每个分支使用哪些平面；同一场景出现多个平面时必须明确提交 `asset_bindings`，页面默认记录“每个分支应用到全部已复核平面”的选择，候选有歧义时进入人工复核。前景遮挡抠像、软切和 3D 网格仍需后续版本。场景数和时长是本地处理预算，超过预算的任务会明确失败，不会截断视频或猜测未处理的镜头。

## Pipeline

```text
[门店 MP4 + 1..10 张分店招牌]
                 │
                 ▼
     [三帧采样 + 视觉候选定位]
                 │
        ┌────────┴────────┐
        ▼                 ▼
 [候选通过验证]       [候选需要确认]
        │                 │
        ▼                 ▼
 [SIFT/RANSAC/LK]   [拖动四角后继续]
        └────────┬────────┘
                 ▼
 [每个分支一条 H.264 MP4，输入有音频时保留音轨]
```

视觉模型只负责提出候选平面。每帧跟踪由本地 OpenCV 完成，模型不会被逐帧调用。RANSAC 内点拟合失败的帧写入 `tracking.jsonl`，该帧输出原画。每条成片在进入完成状态前还会经过 FFprobe 契约检查，确认 H.264、`yuv420p`、源视频尺寸、帧数、帧率和音频策略。模型网关错误、坐标响应不合法或成片契约不满足时，任务不会伪造成片。
失败帧比例超过 25% 时，任务进入 `quality_review`，页面保留输出和失败统计供人工复核，不会把部分替换结果直接标成完成。这个阈值只处理跟踪失效，不代表遮挡已经被分割或修复。

## Quickstart

需要 Python 3.11+、FFmpeg（包含 `ffmpeg` 和 `ffprobe`）以及 OpenCV、NumPy、Pillow、Flask。

```bash
python -m pip install -r requirements.txt
python prototype/franchiseflow/app.py
```

打开 `http://127.0.0.1:5000/`。页面支持动态添加分支，最多 10 张图片，也可以填入品牌名或招牌外观提示帮助视觉定位。任务通过 `POST /api/jobs` 创建，使用 `GET /api/jobs/<job_id>` 查询状态；`GET /api/jobs?status=needs_review` 可以从持久化目录找回待复核任务。候选需要人工确认时，提交四角或 `surfaces[]` 到 `POST /api/jobs/<job_id>/review`，完成后从 `GET /api/jobs/<job_id>/outputs` 获取结果。旧的 `POST /api/render` 仍保留为同步兼容入口，若返回 `needs_review` 会同时给出 `job_id` 和复核地址。服务会拒绝超过 250 MiB 的上传、超过 60 秒的视频或超过 8 个硬切场景的视频。

视频输入支持 `.mp4`、`.mov`、`.m4v`、`.webm`、`.avi` 和 `.mkv`。非 MP4 源以及检测到 HEVC、H.265、ProRes、AV1、未知编码或非 `yuv420p` 像素格式的 MP4 会在任务创建时保留原文件，并用 FFmpeg 生成 `processing_input.mp4`（H.264、`yuv420p`、可选 AAC 音轨）供 OpenCV 和跟踪器使用。任务 JSON 与 `manifest.json` 会记录原始扩展名、工作文件名和解码器可见的尺寸、帧率、时长估计；场景采样阶段再顺序解码并记录实际帧数。

### API key

复制 `.env.example` 为 `.env`，只在本地填写：

```bash
BANANA_API_KEY=...
BANANA_API_BASE_URL=https://keungliang.dpdns.org/v1
BANANA_VISION_MODEL=gemini-3.1-flash-image
```

`.env` 被 Git 忽略。日志、任务 JSON、预览图和提交历史不会写入密钥。

### Reproduce the tracked sample

`demo_cut.mp4` 是由公开素材生成的本地文件，默认不提交到 Git。需要时运行：

```bash
ffmpeg -y -i spikes/franchiseflow/fixtures/real_storefront.mp4 -t 6.0 -c copy spikes/franchiseflow/fixtures/demo_cut.mp4
```

使用人工校准基线运行本地跟踪：

```bash
python spikes/franchiseflow/tracker.py \
  --video spikes/franchiseflow/fixtures/demo_cut.mp4 \
  --quad-json spikes/franchiseflow/quad.json \
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png \
  --out-dir spikes/franchiseflow/outputs_demo_h264
```

`--quad-json` 明确表示人工基线。若要标记为模型或人工复核结果，可传 `--initialization-mode vision_model` 或 `--initialization-mode reviewed_candidate`，不能用参数名推断来源。每个任务目录还会写入不含密钥的 `manifest.json`，记录模型、样本哈希、候选来源和输出状态。

## 实测边界

- 0.30 到 0.36 px 是 RANSAC 单应矩阵的 SIFT 内点重投影拟合残差，不是招牌物理边缘的真实定位误差。
- 20 帧短测记录过 475 个首帧特征、288 个中位内点和 0.30 px 残差；这些数字不能代表整段视频。
- 6 秒、144 帧的人工基线样例曾用约 122 秒完成，其中纯跟踪约 63 秒；具体耗时取决于机器和 FFmpeg。
- 回退只处理已经检测到的跟踪失败，当前没有 SAM 或视频抠像，行人经过招牌时可能被替换画面覆盖。
- 输入有音频时，FFmpeg 使用原视频音轨合成输出；无音频输入不会添加静音轨。
- 当前服务在本地线程中处理任务，任务状态保存为每个任务目录下的 JSON；生产部署还需要队列、权限和清理策略。
- 场景检测采用保守的硬切双阈值；低置信度软切可能被合并到同一场景，当前不会自动猜测新的场景边界。

## Scene / Surface 实验接口

跟踪器已经提供最多四个独立平面的实验参数 `--surface-quad-json` 和 `--asset-binding-json`。Web 任务使用最多八个硬切场景的清单，每个场景最多四个平面。每个平面拥有自己的 SIFT/RANSAC/LK 跟踪器，单个平面失效时只保留该平面的原画，其他平面继续处理。一个分支可以显式引用多个平面，多个分支也可以分别引用同一个平面。软切、遮挡抠像和跨镜头自动匹配仍需单独验证。

## Tests

```bash
python -m unittest discover -s spikes/franchiseflow/tests -v
python -m unittest discover -s prototype/franchiseflow/tests -v
git diff --check
git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'
```

测试覆盖模型错误脱敏、候选四边形校验、三帧验证、硬切场景采样、逐场景跟踪、SIFT/LK 失败帧、FFmpeg 缺失、音频参数、1 到 10 个分支、任务持久化、人工复核和 Range 播放接口。

## Known limitations

真实样例中，`gemini-3.1-flash-image` 可能把倾斜立面的砖墙、窗户或门洞混入四边形。系统会把这种候选交给三帧验证和人工确认，不把人工 `quad.json` 冒充模型定位结果。视觉模型的默认提示词只描述通用的平面招牌；用户可以通过 `target_hint` 传入品牌名或招牌外观描述，避免把演示素材的文字当成通用规则。

## Credits

- Storefront video: Pexels, license record in `spikes/franchiseflow/fixtures/source.md`
- Vision model: `gemini-3.1-flash-image` through the configured OpenAI-compatible gateway
- Core libraries: OpenCV, NumPy, Pillow, Flask, FFmpeg
- License: MIT
