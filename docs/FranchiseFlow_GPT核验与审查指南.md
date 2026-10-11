# FranchiseFlow 给 GPT 的核验与审查说明（含专用 Prompt）

本文档专为交接给 GPT 执行**第三方独立代码审查、技术指标核验与合规审计**而设计。你可以直接复制下方的【GPT 审查专用的输入 Prompt】发送给 GPT，让其对当前工作区和最新提交执行严格审查。

---

## 一、GPT 审查专用 Prompt（可直接复制使用）

```markdown
你现在作为一名严苛的资深计算机视觉（CV）工程师兼开源黑客松项目技术审计专家，对当前的 FranchiseFlow 项目进行全方位的技术审查与验收核验。

### 背景信息
- **项目仓库**：https://github.com/f22363712-dotcom/franchiseflow-image-ai
- **基线提交**：`9bb880c`
- **本次修正提交**：`c587576`（分支：master）
- **核心定位**：移动镜头门店视频 · 首帧视觉大模型初筛定位 · 本地 OpenCV SIFT/RANSAC 几何跟踪 · 三路分店 H.264 视频并行生成
- **黑客松限制**：Devpost 视频严格不超过 3 分钟（180 秒）；严禁代码/日志/提交中泄露 API Key；严禁虚假宣传与过度承诺。

### 你的审查任务清单（Checklist）

请针对以下 6 个核心维度进行严格核验，并逐项给出【通过 / 警告 / 不通过】评级及改进建议：

#### 1. 视觉定位与多模态模型真实性（Zero-BS 真实度审查）
- 检查 `spikes/franchiseflow/vision_client.py` 与 `auto_quad.py`：实际调用的模型是否已统一为授权的 `gemini-3.1-flash-image`？
- 审查自动定位提示词：是否针对真实样例明确了带有 "STATE OF CONFUSION" 的深灰色招牌板，且**绝对没有硬编码**样例的像素坐标？
- 检查项目是否诚实地披露了“多模态大模型在倾斜 3D 立面上缺乏空间几何感知，导致选区斜切砖墙与门窗”的实验事实？
- 检查工作流定位：是否明确采用“AI 视觉初筛 + 人工基线校准（Human-in-the-loop）”架构，没有将人工 `quad.json` 伪装成全自动模型的成果？

#### 2. 浏览器视频播放与 H.264 / faststart 转码管线审查
- 检查 `spikes/franchiseflow/tracker.py` 中的 `transcode_to_h264` 实现：是否在 OpenCV 输出后使用 FFmpeg 强制压制为 `libx264`、`yuv420p`、`-movflags +faststart`？
- 检查环境依赖与报错机制：当系统缺少 FFmpeg 时，是否在 `run_tracker` 启动阶段立即抛出显式清晰的 `RuntimeError`，阻止盲目运算？
- 检查 Web 服务分发：`prototype/franchiseflow/app.py` 中的 API 返回的成片 URL 是否指向转码后的 H.264 文件？是否支持 HTTP 206 Partial Content 分段 Range 拖拽？

#### 3. 性能指标与文案诚实度核实（删除未经验证承诺）
- 检查 `README.md`、`prototype/franchiseflow/templates/index.html`、`static/app.js`：
  - 是否彻底删除了“Zero Jitter”、“100% 完美贴合”、“广播级成片”等浮夸承诺？
  - 是否明确阐明：**0.36 px（及 20 帧短测的 0.30 px）是 RANSAC 单应矩阵的特征内点重投影拟合残差，绝不是招牌物理边缘的真实定位误差**？
  - 是否明确区分了“20 帧短测（475 特征/288 内点）”与“144 帧完整 demo_cut（耗时 122s / 纯跟踪 63s / 113 内点）”的指标边界？
  - 是否坦率披露了系统物理边界：回退策略仅针对检测到的跟踪失败（几何退化/内点不足）保留原画，目前**尚未包含前景遮挡抠像（无 SAM/Matting）**，且**不保留原视频音频**？
  - 是否提供了从公开素材生成 `demo_cut.mp4` 的精确 FFmpeg 命令？

#### 4. 自动化测试套件与覆盖率审查
- 审查 `spikes/franchiseflow/tests/`（15 项测试）与 `prototype/franchiseflow/tests/`（7 项测试）：
  - 是否补齐了 FFmpeg 缺失的单元测试（`test_ffmpeg_missing_raises_runtime_error`）？
  - 是否包含 HTTP 403 过滤、API 超时、几何非凸、越界、单位坐标归一化以及合成运动与失败帧原画回退测试？
  - 是否包含 Web 服务的 `/api/demo_results` 预置样片分发与 400 异常参数拦截？

#### 5. 安全合规与 Git 提交洁净度
- 执行全仓库 Secret Scan：是否存在任何 `sk-`、`Bearer`、真实 API 密钥的硬编码或意外提交？
- 检查 Git 提交：`c587576` 是否纯净仅包含 FranchiseFlow 目录下的文件？是否完全隔离了 RealityLock、AdSafeZone 等旧探索文件？

#### 6. 3 分钟录屏脚本（Demo Script）可行性与节奏审查
- 审查 `docs/FranchiseFlow_Demo脚本.md`：
  - 节奏是否严格符合最新指导：“**先展示原片与三路成片，再解释视觉定位和几何跟踪**”？
  - 视频总时长是否稳定控制在 2 分 45 秒内（留出 15 秒安全冗余，严控在 180 秒内）？
  - 是否提供了无需等待 2 分钟计算、可直接离线回放真实已转码样片的便捷操作动线？

请以结构化的 Markdown 报告格式输出你的审查结论，包含各模块打分、发现的潜在隐患（若有）以及发布录屏前的最终放行建议（Go / No-Go）。
```

---

## 二、项目核心改动速查表（供 GPT 交叉比对）

为帮助 GPT 高效对齐代码实现，以下是本次提交（Commit `c587576`）的核心文件变动速查：

| 文件路径 | 核心变动内容 | 关键代码行/特征 |
| :--- | :--- | :--- |
| `spikes/franchiseflow/tracker.py` | 增加 `transcode_to_h264`，启动时检测 FFmpeg，中间文件写 `_raw_*.mp4` 后转码为 H.264+yuv420p+faststart 并清理临时文件 | `libx264`, `yuv420p`, `+faststart`, `shutil.which("ffmpeg")` |
| `spikes/franchiseflow/tests/test_tracker.py` | 增加 `test_ffmpeg_missing_raises_runtime_error` 单元测试 | `@patch("shutil.which", return_value=None)` |
| `spikes/franchiseflow/vision_client.py` | 默认模型切换为 `gemini-3.1-flash-image`，提示词增加负向几何约束，支持 `target_hint` 入参 | `gemini-3.1-flash-image`, `target_hint` |
| `spikes/franchiseflow/auto_quad.py` | 增加 `--target-hint` 参数透传给 `locate_sign` | `args.target_hint` |
| `prototype/franchiseflow/app.py` | 增加 `/api/demo_results` 接口，分发预置转码完成的 3 路 H.264 视频 | `@app.route("/api/demo_results")` |
| `prototype/franchiseflow/tests/test_app.py` | 增加 `test_get_demo_results` 单元测试，更新模型 Mock 署名 | `test_get_demo_results` |
| `prototype/franchiseflow/templates/index.html` | 修正界面模型徽章与文案，提示词切为人工校准基线，明确残差为拟合残差 | `Gemini-3.1-Flash`, `人工校准基线`, `非边界误差` |
| `prototype/franchiseflow/static/app.js` | 增加一键载入样例 `/api/demo_results`，自动载入 H.264 视频与指标 | `btnLoadSample`, `sample_first_frame.jpg` |
| `README.md` | 统一模型名称，重构架构图，更新 144 帧实测指标与残差定义，增加 `demo_cut.mp4` ffmpeg 生成命令 | 完整复现命令与真实工程边界披露 |
| `docs/FranchiseFlow_Demo脚本.md` | 重构为“成片成果直击 → 定位解构 → 几何跟踪 → 边界与耗时 → 开源复现”，严格限时 2 分 45 秒 | 5 段式严谨台词与画面脚本 |
| `docs/FranchiseFlow_发布前修正与实测报告.md` | 完整记录 4 轮提示词对比实验坐标、FFprobe 视频流与 faststart 字节偏移量、HTTP 206 测试数据 | 详实工程实测证据库 |

---

## 三、现场实测证据与验证命令（可直接提供给 GPT 审查）

### 1. 单元测试验证（22 项全过）
```bash
# Spikes 测试（15 项）
python -m unittest discover -s spikes/franchiseflow/tests -v

# Prototype Web 测试（7 项）
python -m unittest discover -s prototype/franchiseflow/tests -v
```

### 2. 视频编码与 faststart 验证命令
```bash
# 验证 H.264 与 yuv420p
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt -of json spikes/franchiseflow/outputs_demo_h264/branch_1_riverside.mp4

# 验证 faststart (moov atom 置前)
python -c "
with open('spikes/franchiseflow/outputs_demo_h264/branch_1_riverside.mp4', 'rb') as f:
    print('moov offset:', f.read(1024).find(b'moov'))
"
# 输出应为：moov offset: 36
```

### 3. 安全检查命令
```bash
git diff --check
git grep -n -I -E "sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}" -- spikes/franchiseflow prototype/franchiseflow docs/FranchiseFlow* README.md .env.example
# 输出应为空
```

---

## 四、建议的 GPT 审查交付物格式

GPT 审查完毕后，应向你交付包含以下模块的审查报告：
1. **执行摘要（Executive Summary）**：总体评价与放行决定（Go / No-Go）。
2. **逐项审查结果表（Audit Matrix）**：覆盖 Prompt 中的 6 个核心维度。
3. **亮点与技术信誉评估**：如对多模态 3D 透视局限性的坦诚披露、H.264 浏览器直播的完整实现等。
4. **现场录屏避坑指南**：提醒录制时的麦克风、演示节奏、以及如何使用“一键载入样例”节省录屏时间。
