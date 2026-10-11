# FranchiseFlow 项目交接文档 (Handoff for Next Agent / GPT)

> **编写时间**：2026-10-12 01:10（北京时间）  
> **提交截止**：2026-10-12 12:00（Banana Hacks 黑客松）  
> **当前状态**：核心四个阶段任务（Task 1–4）全部实现并通过回归测试，代码已同步推至 GitHub `master` 分支。  
> **文档性质**：这是早期黑客松交接记录。下面的“三张素材、三路全自动”描述只适用于当时的单平面样片，当前产品契约以仓库根目录 `README.md`、任务 `manifest.json` 和页面人工复核状态为准。
> **仓库地址**：[https://github.com/f22363712-dotcom/franchiseflow-image-ai.git](https://github.com/f22363712-dotcom/franchiseflow-image-ai.git)  
> **基线与最新提交**：从 `9982ac3` 出发，当前远程基线为 `d09960b`；本页早期提交号仅作历史参考。

---

## 1. 项目定位与核心问题

### 1.1 业务场景
实体连锁加盟品牌（餐饮、零售、便利店等）通常只有总部拍摄的一条高质量移动实拍视频。当需要为数十家加盟分店制作本地化广告投放素材时，传统后期（AE/PR）需要逐帧打关键帧和透视对齐，耗费数小时。

### 1.2 FranchiseFlow 解决方案
**一条移动门店实拍视频 + 分店招牌图 → 生成多路独立透视贴合 MP4 视频**。候选平面经过验证后自动跟踪，存在歧义时需要人工确认。

### 1.3 核心技术链路（Hybrid AI + Classical CV）
```
[移动实拍视频 MP4] (抽取首帧)
        │
        ▼
[多模态视觉大模型] ───► [招牌物理平面四角 TL, TR, BR, BL]
        │                           │
        └──────────────┬────────────┘
                       ▼
         [本地高精度几何跟踪器]
     - OpenCV SIFT 首帧锚点匹配 (BFMatcher + Ratio Test 0.75)
     - RANSAC 单应矩阵解算 (阈值 3.0px)
     - Lucas-Kanade 双向金字塔光流兜底 (FB-Error < 1.0px)
     - 几何退化/遮挡失败帧安全保底原片 (Fail-Safe)
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
 [分支 1 MP4]    [分支 2 MP4]    [分支 3 MP4]
 (滨河路店)       (市中心店)       (北园店)
```

**为什么是混合架构而不是逐帧大模型？**
1. **避免逐帧模型坐标漂移**：多模态大模型只提出首帧候选，后续帧由固定锚点的几何跟踪器处理，减少独立推理带来的坐标跳动。
2. **极低成本与零排队**：长视频逐帧调用多模态 API 成本极高且受网络波动制约；本方案只需调用 1 次模型，后续全部在本地毫秒级纯数学解算。

---

## 2. 当前代码库资产分布

| 模块 / 路径 | 核心作用 | 关键函数 / 资产 |
| :--- | :--- | :--- |
| `spikes/franchiseflow/vision_client.py` | OpenAI 兼容多模态视觉客户端 | `VisionClient.locate_sign()`、`normalize_quad()`（包含凸多边形/单位/数值防御性校验） |
| `spikes/franchiseflow/auto_quad.py` | 独立 CLI 招牌平面定位探测工具 | 接收 `--image`，输出带有四角/置信度/推理原因的 JSON 与预览图 |
| `spikes/franchiseflow/tracker.py` | 核心平面跟踪与多路合成流水线 | `run_tracker(video, quad, signs, out_dir, ...)`、`PlanarTracker` |
| `spikes/franchiseflow/metrics.py` | 跟踪评价与残差指标计算 | `compute_summary()`（计算内点中位数、重投影误差中位/95分位、失败帧统计） |
| `prototype/franchiseflow/app.py` | 单任务轻量级 Web Demo 后端 (Flask) | `POST /api/render`、`GET /api/jobs/<job_id>`、`GET /outputs/...` |
| `prototype/franchiseflow/templates/index.html` | 前端界面（黑客松深色玻璃拟态美学） | Canvas 招牌四角投影、指标仪表盘、三路视频同步播放卡片 |
| `prototype/franchiseflow/static/app.js`, `app.css` | 前端交互与样式 | 拖拽上传、异步状态轮询、角点可视化标注、播放控制 |
| `spikes/franchiseflow/tests/` | 跟踪算法与视觉客户端单元测试 | 14 项测试：已知单应矩阵测试、退化拦截、403脱敏拦截等 |
| `prototype/franchiseflow/tests/` | Web 端到端单元测试 | 覆盖参数校验、网关异常拦截、Mock 流水线、人工复核和任务恢复 |
| `spikes/franchiseflow/fixtures/` | 真实基准视频与剪辑版本 | `real_storefront.mp4` (Pexels 9.7s)、`demo_cut.mp4` (6.0s无遮挡版) |
| `docs/` | 文档与交付件 | `FranchiseFlow_Demo脚本.md`（<=3分钟录屏脚本）、`FranchiseFlow_API接入记录.md`、`FranchiseFlow_代码审查报告.md` |

---

## 3. API 网关实测现状与结论

在 `D:\Obsidian\obsidian_kb\20_Areas\保存\Banana Hacks api.md` 中记录了两个网关，实测状态如下：

1. **网关 1：`https://elysia.h-e.top` / `https://e.c.h-e.top`**  
   - 状态：**HTTP 403 Forbidden**（持续存在，疑为服务停用或严格来源白名单）。
2. **网关 2：`https://keungliang.dpdns.org/v1`**  
   - 状态：**HTTP 200 OK**（需在 Request Header 中注入标准浏览器 `User-Agent`，已在 `vision_client.py` 中内置适配）。
   - 挂载模型：51 款，包含 `gemini-3.1-flash-image`、`glm-5.3-flash` 等。
   - **实测结果**：使用 `gemini-3.1-flash-image` 对 `real_storefront_first.jpg` 成功输出高精度的四角坐标 `[[0, 103], [760, 497], [803, 706], [0, 240]]` 与空间几何推理理由。
   - 代入真实视频：首帧提取 SIFT 点 **475 个**，中位内点 **288 个**，重投影残差中位数 **0.30 px**，**0 失败帧**！
3. **离线兜底机制（Manual Fallback）**：  
  - 当完全无网络或云端超时，支持读取 `quad.json`（带有显式 `manual_fallback` 标签），用于离线人工基线演示。

---

## 4. 当前运行方式

### 4.1 启动 Web Demo
```powershell
python prototype/franchiseflow/app.py
```
- 访问：`http://127.0.0.1:5000/`
- 输入：
  - 视频：`spikes/franchiseflow/fixtures/real_storefront.mp4`（或 `demo_cut.mp4`）
  - 分支素材：`spikes/franchiseflow/assets/` 下的三张图片（`riverside.png`、`downtown.png`、`northpark.png`）
- 模式：可选“视觉模型模式”或“手动基线兜底”。任务列表可通过 `GET /api/jobs?status=needs_review` 找回待复核作业。
- 断网多镜头任务可在 `POST /api/jobs` 传入 `manual_scene_quads`，按检测场景顺序提供四角。每个场景仍需通过多帧纹理与单应性验证，长度不符或验证失败会停在 `needs_review`。
- 可通过页面的“保存为品牌配置”面板或 `POST /api/brand-profiles` 保存品牌提示词、分店标签和招牌图片；后续任务只传 `brand_profile_id` 和视频即可复用，任务清单会记录配置 ID、素材 SHA-256 和素材适配策略。配置默认用 `surface_policy=review`，确认所有分店素材都覆盖全部已验证平面时才使用 `surface_policy=all_surfaces`。`scene_reuse_policy=reuse_if_verified` 会尝试跨镜头复用候选，复用必须通过当前镜头的 SIFT/RANSAC 和多帧验证。
- 自动定位会把任务中的前四张分店素材作为品牌外观参考图发送给视觉模型，提示词要求四角坐标只能来自视频联系图；参考图 SHA-256 会写入视觉审计。它只提供外观线索，不能替代多帧几何验证。
- 页面新增“批量本地化生产”区域。选择已保存的品牌配置后可以一次提交最多 8 条视频，页面轮询批次状态并展示每条成片链接；单条任务需要人工复核时，可从批次卡片打开原有复核面板。批次也可以通过 `POST /api/batches` 和 `GET /api/batches/<batch_id>` 使用。
- 批次中的 `failed` 或 `vision_failed` 子任务会显示“重试此视频”。重试沿用原任务 ID、视频和分店素材，清除旧输出后重新进入队列；队列满或存储异常会恢复原失败状态和产物。
- 批次交付可以调用 `GET /api/batches/<batch_id>/download` 获取 ZIP。根目录 `batch_manifest.json` 列出所有子任务及状态；只有已完成且文件路径校验通过的任务才会包含 `jobs/<job_id>/manifest.json`、MP4 和 SHA-256/字节数清单。未完成、失败、人工复核或缺少成片的任务只保留状态，不会进入视频目录；没有任何可交付子任务时接口返回 409。
- 服务重启恢复时会扫描带有 `batch_id` 的持久化子任务，把进程在 `job.json` 已落盘、批次索引尚未追加的窗口中留下的子任务补回 `job_ids`。没有 `batch_id` 的普通单任务，以及批次文件已经删除的任务，不会被误加入批次。
- `POST /api/jobs` 和 `POST /api/batches` 支持可选的 `Idempotency-Key` 请求头。同一接口、同一 key 和相同表单/文件内容会返回原来的资源，不会创建重复任务；同一接口复用 key 但请求内容变化会返回 HTTP 409。不带该请求头时保持原有行为。key、请求指纹和资源 ID 写入 `prototype/franchiseflow/outputs/idempotency.json`，原子替换保存，服务重启后仍可恢复；当前实现使用单进程锁，多副本部署需换共享存储与分布式锁。

### 4.2 运行全量自动化测试
```powershell
python -m unittest discover -s spikes/franchiseflow/tests -v
python -m unittest discover -s prototype/franchiseflow/tests -v
```
（当前两套测试均已通过，具体数量以命令输出为准）

### 4.3 零泄密与格式检查
```powershell
git diff --check
git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'
```

---

## 5. 现存已知限制与技术权衡（Honest Disclosures）

1. **前景物体遮挡（Occlusion Handling）**：
   - 在 9.7 秒完整视频的第 155 至 161 帧（约 3.0%），因行人前景横穿招牌，导致特征点被遮挡。
   - 当前处理方式：高置信度局部残差会保留原始前景像素；跟踪失效帧继续使用**安全回退保留原画（Fail-Safe）**，不渲染悬浮错位贴图。
   - 这层保护依赖平面配准和局部残差，不能替代 SAM 或通用视频抠像，摘要会记录 `occlusion_hints` 供复核。
   - 现成方案：已准备 6.0 秒干净运镜版本 `spikes/franchiseflow/fixtures/demo_cut.mp4`（144 帧），用于演示无遮挡运镜；完整片段作为压力测试保留。
2. **CPU 渲染时长**：
   - 720p 完整 233 帧逐帧进行 SIFT 特征检测与单应变换在纯 CPU 上耗时约 60~100 秒。Web 界面有 Loading 状态展示，适合单次异步渲染，不适合毫秒级实时交互。

---

## 6. 接棒 Agent / GPT 的审查重点与后续方案决策

当前距离提交截止（2026-10-12 12:00）时间非常宝贵，接棒者应当**避免无意义的重构或扩大范围（Scope Creep）**，聚焦于以下决策：

### 决策点 1：下一步重点是“工程精进”还是“准备最终提交材料与录屏”？
- **选项 A（强烈建议）：转向演示录屏与材料打磨**
  - 当前代码已经具备：在线视觉大模型候选定位、SIFT 几何跟踪、多路视频渲染导出、任务状态与人工复核页面、回归测试和 <= 3 分钟录屏脚本。多场景、多平面任务需要提交明确的 `asset_bindings`。
  - 接下来的最大产出是：指导用户完成 1080p 录屏（不超过 180 秒）、上传 YouTube/Bilibili、撰写 Devpost 提交描述（解决的痛点、技术链路、创新性、署名）。
- **选项 B：微调级算法视觉增强（若时间充裕）**
  - 是否需要在合成边缘增加 1~2 像素高斯模糊羽化（Feathering）以进一步提升招牌融合自然度？
  - 是否在 Web 界面增加对 `gemini-3.1-flash-image` 的显式下拉模型切换支持？

### 决策点 2：演示视频的展示策略
- 脚本已在 [docs/FranchiseFlow_Demo脚本.md](docs/FranchiseFlow_Demo脚本.md) 定义好严格的 2分55秒 流程。
- 接棒者应协助用户梳理录制步骤，并提醒：
  1. 绝不在录屏中暴露任何 API 密钥或 `.env` 内容；
  2. 演示中明确展示“首帧视觉模型识别平面 + 后续帧本地几何跟踪”的技术分工，体现针对黑客松题目的深度思考。

---

## 7. Git 提交记录参考

```
1d3ae19 docs: add FranchiseFlow code review report
ec1a57e docs: record successful validation with alternative vision gateway
779b31c fix: add browser User-Agent to vision client and ignore auto quad outputs
3838dad docs: prepare FranchiseFlow submission
bb8ebdd feat: add FranchiseFlow demo page
9cee39c feat: expose safe planar tracking pipeline
640324b test: lock vision gateway contract
9982ac3 docs: tighten secret scan (基线)
```
所有修改已全部推送到远程仓库，随时可进行拉取、审查或在此基础上继续推进！
