# FranchiseFlow

**一条移动门店视频，生成 1 到 10 条分店本地化成片。**

FranchiseFlow 面向连锁品牌的视频本地化制作。用户上传一条连续门店运镜视频和若干分店招牌图片，系统先让视觉模型提出平面候选，再用三帧检查判断候选能否交给本地几何跟踪器。候选不可靠时，页面展示原因并允许拖动四个角点确认。

当前产品原型支持 720p、60 秒以内的视频，自动识别最多 8 个高置信度硬切场景，并对满足保守统计条件的简单 dissolve 建立新场景锚点；每个场景最多处理 4 个经过验证且互不重叠的平面，分支数量为 1 到 10 个。多平面任务会记录每个分支使用哪些平面；同一场景出现多个平面时必须明确提交 `asset_bindings`，页面默认记录“每个分支应用到全部已复核平面”的选择，候选有歧义时进入人工复核。系统会对高置信度的局部前景残差执行前景遮挡保护，保留原始像素，但这不是语义分割或通用视频抠像；复杂转场和 3D 网格仍需后续版本。场景数和时长是本地处理预算，超过预算的任务会明确失败，不会截断视频或猜测未处理的镜头。

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
 [每个分支一条带分店标签的 H.264 MP4，输入有音频时保留音轨]
```

视觉模型只负责提出候选平面。每帧跟踪由本地 OpenCV 完成，模型不会被逐帧调用。RANSAC 内点拟合失败的帧写入 `tracking.jsonl`，该帧输出原画。每条成片在进入完成状态前还会经过 FFprobe 契约检查，确认 H.264、`yuv420p`、源视频尺寸、帧数、帧率和音频策略。模型网关错误、坐标响应不合法或成片契约不满足时，任务不会伪造成片。
失败帧比例超过 25% 时，任务进入 `quality_review`，页面保留输出和失败统计供人工复核，不会把部分替换结果直接标成完成。每条通过契约检查的成片还记录 SHA-256，方便下载后核对文件身份。遮挡保护帧会写入 `occlusion_hints`，供复核时检查。

## Quickstart

需要 Python 3.11+、FFmpeg（包含 `ffmpeg` 和 `ffprobe`）以及 OpenCV、NumPy、Pillow、Flask。

```bash
python -m pip install -r requirements.txt
python prototype/franchiseflow/app.py
```

打开 `http://127.0.0.1:5000/`。页面支持动态添加分支，最多 10 张图片，也可以填入品牌名或招牌外观提示帮助视觉定位。页面会读取本地品牌配置列表，选择配置后可以直接复用分店素材；也可以在“保存为品牌配置”面板中一次提交当前图片、品牌提示、分支标签和三项策略，保存成功后页面会自动选中新的配置 ID。页面的“批量本地化生产”区域支持选择一个品牌配置后一次提交最多 8 条视频，并轮询每条视频的状态、输出和人工复核入口。分店标签默认取上传文件名，也可以在 `POST /api/jobs` 的 `branch_labels` 字段提交 JSON 数组；标签会写入任务清单并进入输出文件名，避免批量交付时只看到 `branch_1`。任务通过 `POST /api/jobs` 创建，使用 `GET /api/jobs/<job_id>` 查询状态；`GET /api/jobs?status=needs_review` 可以从持久化目录找回待复核任务。候选需要人工确认时，提交四角或 `surfaces[]` 到 `POST /api/jobs/<job_id>/review`，完成后从 `GET /api/jobs/<job_id>/outputs` 获取结果。跟踪质量进入 `quality_review` 时，页面会播放已生成成片；人工决定后向 `POST /api/jobs/<job_id>/quality-review` 提交 `{"decision":"accept"}` 或 `{"decision":"reject"}`。旧的 `POST /api/render` 仍保留为同步兼容入口，若返回 `needs_review` 会同时给出 `job_id` 和复核地址。服务会拒绝超过 250 MiB 的上传、超过 60 秒的视频或超过 8 个硬切场景的视频。

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
- 跟踪失败帧会保留原画；高置信度局部前景残差会触发原始像素保护。系统没有 SAM 或通用视频抠像，复杂遮挡仍可能需要人工复核。
- 输入有音频时，FFmpeg 使用原视频音轨合成输出；无音频输入不会添加静音轨。
- 当前服务在本地线程中处理任务，任务状态保存为每个任务目录下的 JSON；生产部署还需要队列、权限和清理策略。
- 场景检测采用保守的硬切和短窗口 dissolve 判断；低置信度复杂转场可能被合并到同一场景并进入复核。
- 断网时可以向 `POST /api/jobs` 提交 `use_manual_quad=true` 和可选的 `manual_scene_quads` JSON 数组。数组按检测出的场景顺序提供四角，系统逐场景运行多帧纹理与单应性验证。数组长度不匹配或任一场景验证失败会停在 `needs_review`，不会把首个镜头的坐标复制到整段视频。

## Scene / Surface 实验接口

跟踪器已经提供最多四个独立平面的实验参数 `--surface-quad-json` 和 `--asset-binding-json`。Web 任务使用最多八个场景的清单，每个场景最多四个平面。每个平面拥有自己的 SIFT/RANSAC/LK 跟踪器，单个平面失效时只保留该平面的原画，其他平面继续处理。一个分支可以显式引用多个平面，多个分支也可以分别引用同一个平面。场景清单会记录 `transition_type`，目前 `dissolve` 只覆盖短窗口、全画面颜色变化明显的简单转场，复杂转场仍可能合并到同一场景并进入质量复核。`asset_bindings.fit` 支持 `cover` 和 `contain`，前者铺满平面，后者保留素材宽高比并在平面内留出透明边界。高置信度遮挡提示会保留原始前景像素并写入摘要，系统仍不声称具备通用视频抠像或跨镜头自动匹配能力。

## Brand profiles

重复使用同一品牌的分店招牌时，可以先创建本地品牌配置。配置只保存品牌提示词、分店标签和图片副本，不保存 API key，也不把本机绝对路径写进响应。

```bash
curl -X POST http://127.0.0.1:5000/api/brand-profiles \
  -F 'profile_id=north-star' \
  -F 'brand_name=North Star Coffee' \
  -F 'target_hint=dark charcoal fascia with white wordmark' \
  -F 'surface_policy=all_surfaces' \
  -F 'scene_reuse_policy=reuse_if_verified' \
  -F 'branch_labels=["滨河路店","市中心店"]' \
  -F 'branches=@assets/riverside.png' \
  -F 'branches=@assets/downtown.png'
```

创建任务时只上传视频并传 `brand_profile_id=north-star`，系统会复制配置中的图片到任务目录，并沿用标签和 `target_hint`。配置和任务清单会保存每张素材的 SHA-256 与文件大小，便于追踪实际使用的图片版本。一次任务可以继续直接上传 `branches`，两种方式不能混用，避免分店顺序产生歧义。配置目录默认在 `prototype/franchiseflow/brand_profiles/`，已加入 Git 忽略规则。

自动定位时，任务会把前四张分店素材作为可选的品牌外观参考图，与视频联系图一起发送给视觉模型。提示词明确参考图只用于判断品牌外观，四角坐标只能来自视频画面；每张参考图的 SHA-256 会写入视觉审计，便于确认模型实际看到的素材版本。参考图不能替代多帧几何验证，模型仍可能返回需要人工复核的候选。

配置还可以带 `default_fit=contain`。当每个场景只有一个通过验证的平面时，系统会自动把这个素材适配策略写入每个分支。多平面场景默认使用 `surface_policy=review` 并要求显式 `asset_bindings`；确认每个分店素材都应覆盖所有招牌平面时，才设置 `surface_policy=all_surfaces`。

`scene_reuse_policy` 默认是 `vision_each_scene`。设置为 `reuse_if_verified` 后，后续镜头会先用 SIFT/RANSAC 把上一镜已验证平面投影到当前镜头，再跑当前镜头的多帧验证；任一条件不满足就调用视觉模型重新定位，并在审计记录中标注回退原因。

同一品牌需要处理多条门店视频时，可以把已保存的品牌配置复用到一个批次。批次仍按单任务保存每条视频的输入、审计与输出，单条任务需要人工复核时不会被其他视频的结果覆盖：

```bash
curl -X POST http://127.0.0.1:5000/api/batches \
  -F 'brand_profile_id=north-star' \
  -F 'videos=@shoot_a.mp4' \
  -F 'videos=@shoot_b.mp4'
```

响应会返回 `batch_id` 和每条视频的 `job_id`。使用 `GET /api/batches/<batch_id>` 查看批次状态，或用 `GET /api/batches?limit=20` 找回最近批次；使用原有的 `GET /api/jobs/<job_id>`、`/review` 与 `/outputs` 查看和处理单条任务。批次一次最多接收 8 条视频；本地 worker 忙时，多出的任务保持 `queued`，等待已有任务结束后自动排队。

失败的子任务可以保留原上传素材直接重试，不需要重新上传视频或分店图片：

```bash
curl -X POST http://127.0.0.1:5000/api/jobs/<job_id>/retry
```

接口只接受 `failed` 或 `vision_failed` 状态。重试成功后沿用原 `job_id` 并重新进入 `queued`；队列满、输入文件缺失或本地存储失败会返回错误，同时保留原失败记录和已有产物。批次页面会在这两种失败状态下显示“重试此视频”。

批次索引采用原子 JSON 写入。Windows 文件短暂占用时会自动重试；本地存储不可用时返回 HTTP 507，并清理已创建的临时子任务。

服务重启时会扫描带有 `batch_id` 的持久化子任务，并把因进程中断而尚未写入批次索引的子任务补回。普通单任务没有 `batch_id`，不会被加入任何批次；批次文件缺失的任务也会保留为独立任务，等待人工处理。

需要让网络重试安全时，可以给 `POST /api/jobs` 或 `POST /api/batches` 加上 `Idempotency-Key` 请求头。相同接口、相同 key 和相同上传内容会返回原来的 `job_id` 或 `batch_id`，不会再次创建任务；同一接口复用 key 但上传内容或表单字段变化会返回 HTTP 409。映射保存在 `prototype/franchiseflow/outputs/idempotency.json`，采用临时文件加原子替换写入，服务重启后仍可识别已提交资源。不传请求头时保持原有行为。当前实现使用本地单进程锁，适合单机 worker；多进程或多副本部署需要换成共享存储和分布式锁。

批次可以下载可审计的 ZIP 交付包：

```bash
curl -L -o franchiseflow-batch.zip \
  http://127.0.0.1:5000/api/batches/<batch_id>/download
```

包内的 `batch_manifest.json` 会列出每个子任务的状态。只有状态为 `completed` 且输出文件仍通过目录校验的任务才会带入 `jobs/<job_id>/outputs/*.mp4`；对应的 `jobs/<job_id>/manifest.json` 保留定位模型、候选验证、跟踪质量和素材哈希等审计字段。排队、人工复核、失败或缺少文件的子任务只写入批次清单，不会被伪装成成片。批次尚未产生任何安全可下载的成片时，接口返回 HTTP 409。

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
