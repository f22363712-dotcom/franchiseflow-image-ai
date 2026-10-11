# FranchiseFlow 发布前修正与实测报告

本文档记录 FranchiseFlow 项目在发布前修正周期（Pre-Release Polish Cycle）中的全部实测数据、工程变更与验收结论。

---

## 一、自动定位视觉接口实测与人机协同方案

### 1. 接口与模型配置
- **授权接口地址**：`https://keungliang.dpdns.org/v1`
- **实测视觉模型**：`gemini-3.1-flash-image`
- **输入样例**：`spikes/franchiseflow/fixtures/real_storefront_first.jpg`（1280×720 分辨率）

### 2. 提示词迭代与模型输出对比
针对首帧立面招牌（深灰色招牌板，带有 “STATE OF CONFUSION” 字样），设计并实测了 4 轮提示词对比实验（无密钥记录）：

| 提示词版本 | 提示词策略说明 | 返回四角坐标 (TL, TR, BR, BL) | 置信度 | 实际视觉选面分析 |
| :--- | :--- | :--- | :--- | :--- |
| **P1: 基线通用** | 寻找最大的商铺招牌平面 | `[[0, 103], [760, 497], [803, 706], [0, 240]]` | 0.85 | 选区跨越大半个建筑正面，左下角延伸至门洞和街道 |
| **P2: 明确目标名** | 明确指定带有 "STATE OF CONFUSION" 的深灰色招牌板 | `[[303, 102], [760, 471], [724, 617], [299, 290]]` | 0.90 | 选区仍然自左向右严重倾斜，下边缘斜穿红砖墙和 301 A 门窗 |
| **P3: 负向几何约束** | 严禁包含砖墙、窗户、门牌与街道 | `[[303, 102], [760, 471], [724, 617], [299, 290]]` | 0.88 | 模型输出未收敛，依然无法分离斜向拍摄的立面透视 |
| **P4: 锚点结构强化** | 强调查找水平招牌长方形外框 | `[[300, 100], [760, 470], [720, 620], [300, 290]]` | 0.85 | 依然呈下垂斜角 |

### 3. 工程分析与定位策略决定
- **失效原因诊断**：当前通用的多模态大模型在倾斜机位（Oblique Viewpoint）实拍的商业街景中，虽然具备强大的 OCR 和语义识别能力，但**缺乏物理 3D 透视平面的空间几何感知**，极易将沿视线透视缩短的招牌板与下方的红砖墙混淆为一个连贯的大立面，从而产生穿透门窗的倾斜多边形。
- **30 分钟时限决策**：恪守黑客松时限与工程诚信原则，不把该样例坐标硬编码到自动提示词中伪造全自动成功；系统采用已有的**人工校准基线（Manual Calibration Baseline）**：
  - 首帧坐标：`[[220, 55], [900, 300], [910, 410], [220, 165]]`（存储于 `spikes/franchiseflow/quad.json`）
  - 定位模式：明确标注为 `manual_fallback` / `人工校准基线`，供评委审查。
  - 保留模型真实输出供对比：保留 `quad_auto_preview.jpg` 供公开展示，作为多模态大模型在 3D 空间理解局限性的典型真实案例。

---

## 二、浏览器端 H.264 视频真实播放与流式传输验证

### 1. 问题根源
原 OpenCV 默认使用 `VideoWriter_fourcc(*"mp4v")` 导出 MP4 文件，其底层编码为 **MPEG-4 Part 2 (FourCC: FMP4/mp4v)**。该编码格式在 Chrome、Edge、Safari 等主流现代浏览器的 HTML5 `<video>` 标签中无法硬解或直接播放，会触发 `DEMUXER_ERROR_NO_SUPPORTED_STREAMS` 错误。

### 2. 转码管线改造
在 `spikes/franchiseflow/tracker.py` 中引入自动 H.264 转码函数 `transcode_to_h264`：
- **目标编码器**：`libx264`
- **像素格式**：`yuv420p`（浏览器硬件解码通用格式）
- **流式标志**：`-movflags +faststart`（将 moov 索引 atom 移至文件头部）
- **压制参数**：`-preset fast -crf 22 -an`（去除音频，保证高效编码）
- **容错校验**：在 `run_tracker` 启动首行增加 `shutil.which("ffmpeg")` 检测，若系统缺少 FFmpeg 则直接抛出显式清晰的 `RuntimeError`，阻止盲目耗时计算。

### 3. 实测数据与 faststart 检验
对完整 144 帧成片运行 FFprobe 与二进制文件流检验：
```json
{
  "codec_name": "h264",
  "width": 1280,
  "height": 720,
  "pix_fmt": "yuv420p",
  "format_name": "mov,mp4,m4a,3gp,3g2,mj2"
}
```
- **moov Atom 偏移**：字节位置为 `36`，证明 faststart 启用成功，无需下载全片即可瞬间缓冲首帧。
- **文件体积压缩**：原片约 5.8 MB，经过 CRF 22 压制后降至约 1.6 MB，兼具高清画质与秒开速度。

### 4. Flask HTTP 206 Partial Content 分段拖拽实测
在本地运行的 Flask 服务（`http://127.0.0.1:5000`）上验证 HTTP 协议级 Range 请求：
- `GET /outputs/prebuilt_demo/branch_1_riverside.mp4` 返回 `MIME: video/mp4`，文件总长 1,687,212 字节。
- 发送字节范围请求 `Range: bytes=1048576-2097151`，服务响应 `HTTP 206 Partial Content`，返回 `Content-Range: bytes 1048576-1687211/1687212`。
- 证明浏览器在拖拽视频进度条时，能够进行亚秒级 Range Seek，避免卡顿或缓冲失败。

---

## 三、文案统一与证据核实

为防止过度宣传并建立严谨的黑客松技术信誉，全项目完成以下文案与证据修正：

1. **剔除未经验证承诺**：
   - 彻底删除 “Zero Jitter”、“100% 完美贴合”、“广播级成片”、“100% 贴合成功率” 等词汇。
2. **严谨定义残差指标**：
   - 明确指出此前出现的 **0.30 px** 和当前测出的 **0.362 px** 均为**内部 SIFT 特征内点重投影拟合残差（Inlier Reprojection Residual）**，是单应矩阵拟合特征点集的几何一致性指标，**绝不代表招牌物理边界的绝对定位误差**。
3. **区分测试集规模**：
   - 明确指出 “475 特征、288 内点、0 失败” 仅源自初期的 20 帧短视频快测。
   - 完整 144 帧（6.0 秒）运行基于人工校准基线 `quad.json`，耗时 122.08 秒（含 H.264 压制与 faststart 转码，纯跟踪耗时 63.39 秒），中位特征残差 0.362 px，中位 SIFT 内点 113 个。不能仅凭 `--quad-json` 参数判断坐标来自模型。
4. **诚实披露系统边界**：
   - 明确回退策略仅针对**已检测到的几何跟踪失败**（如内点不足或几何退化）触发原画保底；
   - 系统当前**尚未包含前景遮挡语义分割**（无 SAM/Matting，因此前景电线杆穿过时贴图不会被遮挡）；
   - 系统**不保留原视频音轨**（输出静音 MP4）。
5. **提供 `demo_cut.mp4` 准确复现命令**：
   原片来源为 Pexels 公开素材（见 `spikes/franchiseflow/fixtures/source.md`），截取 6.0 秒基准片段命令如下：
   ```bash
   ffmpeg -y -i spikes/franchiseflow/fixtures/real_storefront.mp4 -t 6.0 -c copy spikes/franchiseflow/fixtures/demo_cut.mp4
   ```

---

## 四、抽查帧画面与成片跟随验证

对完整 144 帧生成的 `branch_1_riverside.mp4` 抽查以下关键帧：

1. **首帧（Frame 000）**：
   - 招牌平面四角严密贴合深灰色招牌板，"NOVA COFFEE RIVERSIDE" 文字与边缘视角平滑对齐。
2. **中间帧（Frame 072）**：
   - 镜头向右侧推移并产生轻微透视仰角，单应矩阵连续更新，招牌透视比例随镜头自适应变形，无脱靶漂移。
3. **末帧（Frame 143）**：
   - 镜头平移至最右侧，前景电线杆滑入画面中心，招牌保持精准贴附在背景门头上，证实单应性平滑跟踪全程有效。

---

## 五、完整测试套件与代码审查状态

### 1. 自动化单元测试（22 项全部通过）
- **Spikes 测试**（15 项，耗时 2.9s）：
  - `test_ffmpeg_missing_raises_runtime_error`（验证缺少 FFmpeg 时的明确拦截）
  - `test_run_tracker_synthetic_motion_and_failed_frames`（验证合成运动与失败帧原画回退）
  - `test_run_tracker_validates_inputs`（验证入参边界）
  - `test_vision_client_*`（12 项涵盖 HTTP 403 过滤、超时、非凸、越界、单位坐标归一化等）
- **Web 原型测试**（7 项，耗时 0.8s）：
  - `test_get_demo_results`（验证预置样片查询与转码视频分发）
  - `test_index_page_returns_200`
  - `test_missing_video_returns_400`
  - `test_fewer_than_three_branches_returns_400`
  - `test_unsupported_extension_returns_400`
  - `test_vision_api_failure_returns_error_without_fake_result`
  - `test_successful_render_with_mocked_pipeline`

### 2. 密钥与代码清洁度审查
- 执行 `git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}'`：结果为 0。
- 执行 `git diff --check`：无空白字符或冲突残留。

---

## 六、录屏准备与交付说明

录屏脚本已更新至 [`docs/FranchiseFlow_Demo脚本.md`](file:///c:/Users/20651/Documents/ChatGPT/%E9%BB%91%E5%AE%A2%E6%9D%BE/docs/FranchiseFlow_Demo%E8%84%9A%E6%9C%AC.md)。录屏流程严格控制在 2 分 45 秒内，包含：
1. **0:00 – 0:40**：开篇直接展示原片与三路已转码 H.264 视频在浏览器中的同步播放与拖拽进度；
2. **0:40 – 1:20**：展示视觉大模型初筛定位与人工校准基线（Human-in-the-loop）；
3. **1:20 – 2:00**：解释 SIFT 锚点提取与 RANSAC 单应矩阵亚像素跟踪；
4. **2:00 – 2:35**：展示真实指标仪表盘，说明 0.36 px 拟合残差定义与已知系统边界；
5. **2:35 – 2:50**：展示开源仓库与 ffmpeg 样片复现命令。
