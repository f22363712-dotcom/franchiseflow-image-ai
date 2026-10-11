# FranchiseFlow 实施与交付报告

**交付时间**：2026-10-12
**分支**：master
**基线提交**：`9982ac3`
**代码仓库**：[franchiseflow-image-ai](https://github.com/f22363712-dotcom/franchiseflow-image-ai)

---

## 1. 实际完成的 Task 概览

| 任务 | 核心交付内容 | 对应 Commit Hash | 状态 |
| :--- | :--- | :--- | :--- |
| **Task 1: 验证 GLM-5.3-flash 视觉接口** | 完善 `vision_client.py` 与 `auto_quad.py` 错误拦截；补齐 HTTP 403、超时、`quad: null` 及非凸/越界几何单测；实测网关并记录真实状态 | [`640324b`](https://github.com/f22363712-dotcom/franchiseflow-image-ai/commit/640324b) | ✅ 完成 |
| **Task 2: 整理安全的平面跟踪流程** | 提取 `run_tracker(...)` 统一调用接口；接入 `metrics.py` 统计残差与内点；增加合成运动、单帧失活安全保底与三路成片一致性单测；裁切准备 6s 无遮挡 Demo Cut | [`9cee39c`](https://github.com/f22363712-dotcom/franchiseflow-image-ai/commit/9cee39c) | ✅ 完成 |
| **Task 3: 加入最小本地 Demo 页面** | 创建 `prototype/franchiseflow/` 单任务 Flask 应用；实现素材上传、首帧招牌四角 Canvas 投影高亮、指标仪表盘与三路 MP4 播放器卡片；通过 mock 与真实测试 | [`bb8ebdd`](https://github.com/f22363712-dotcom/franchiseflow-image-ai/commit/bb8ebdd) | ✅ 完成 |
| **Task 4: 准备最终提交材料** | 撰写严格控制在 3 分钟内的 [docs/FranchiseFlow_Demo脚本.md](docs/FranchiseFlow_Demo脚本.md)；全面更新 [README.md](README.md) 比赛清单；执行代码规范扫描与全量回归测试 | [`3838dad`](https://github.com/f22363712-dotcom/franchiseflow-image-ai/commit/3838dad) | ✅ 完成 |

---

## 2. 提交历史（Commit Hashes）

1. `640324b` `test: lock vision gateway contract`
2. `9cee39c` `feat: expose safe planar tracking pipeline`
3. `bb8ebdd` `feat: add FranchiseFlow demo page`
4. `3838dad` `docs: prepare FranchiseFlow submission`

均已全部推送至远程分支 `origin/master`。

---

## 3. 测试命令与执行结果

### 单元测试与端到端测试
运行命令：
```powershell
python -m unittest discover -s spikes/franchiseflow/tests -v
python -m unittest discover -s prototype/franchiseflow/tests -v
```

执行结果：
- **`spikes/franchiseflow/tests`**：14/14 通过（耗时 ~1.6s）
  - 覆盖 fenced JSON、单位坐标归一化、非法越界/极小面积/非凸/非数值坐标拦截
  - 覆盖 HTTP 403 错误遮蔽（无密钥或 Header 泄漏）、超时异常转换、`quad: null` 语义提取
  - 覆盖合成已知单应矩阵误差校验（<3 px）、特征丢失帧安全回退保底原片（`unmodified_original_frame`）、三路输出视频参数一致性校验
- **`prototype/franchiseflow/tests`**：6/6 通过（耗时 ~0.7s）
  - 覆盖主页加载（HTTP 200）
  - 覆盖视频缺失（HTTP 400）、分支数量不足（HTTP 400）、非法后缀拦截（HTTP 400）
  - 覆盖视觉网关异常（HTTP 502，不伪造假结果）
  - 覆盖 Mocked 完整渲染流水线与 `GET /api/jobs/<job_id>` 查询

### 代码质量与安全扫描
- `git diff --check`：0 告警，无行尾空白符或换行格式异常。
- 密钥泄露扫描：
  ```powershell
  git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}' -- ':!D*'
  ```
  结果：0 匹配，无任何明文 Key 或 Token 留存。

---

## 4. Demo 页面启动说明

### 本地 Web 界面
1. 启动服务：
   ```powershell
   python prototype/franchiseflow/app.py
   ```
2. 浏览器访问：`http://127.0.0.1:5000/`
3. 界面操作：
   - 拖拽上传移动镜头视频：`spikes/franchiseflow/fixtures/real_storefront.mp4`（或 6 秒精简剪辑版 `demo_cut.mp4`）
   - 依次上传三张分支设计图：`spikes/franchiseflow/assets/riverside.png`、`downtown.png`、`northpark.png`
   - 初始化模式选择：
     - 若当前网络可连通 GLM 接口，选择 `GLM-5.3-flash 视觉模型`
     - 若当前网络处于 403 或无公网状态，切换为 `手动基线兜底 (Manual Fallback)`
   - 点击“开始跟踪与分支合成”
   - 界面实时显示：识别出的四角坐标投影（Canvas 绘制 TL/TR/BR/BL）、跟踪耗时与 SIFT 匹配指标、三张卡片同步播放渲染后的 MP4 视频并提供下载。

### CLI 命令行备用方案
```powershell
python spikes/franchiseflow/tracker.py `
  --video spikes/franchiseflow/fixtures/real_storefront.mp4 `
  --quad spikes/franchiseflow/quad.json `
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png `
  --out-dir spikes/franchiseflow/outputs2
```

---

## 5. API 状态真实记录：是否仍然 403？

**是的，直接请求官方配置网关仍然返回 HTTP 403。**

实测记录：
- 在本机环境测试 `https://e.c.h-e.top/v1/chat/completions`，网关返回 `HTTP 403 Forbidden`。
- 客户端严格遵循合同，捕获后抛出 `vision gateway returned HTTP 403; check base URL, key, and gateway access`，未生成虚假的 `quad_auto.json`，未在日志泄漏鉴权信息。
- 触发计划设定的止损门禁：**不伪造模型成功，如实记录在 [docs/FranchiseFlow_API接入记录.md](docs/FranchiseFlow_API接入记录.md)**。
- 演示与本地运行提供显式的“手动基线兜底”模式（Manual Engineering Fallback），并在元数据和界面中明确标注，不假冒 AI 在线识别。

---

## 6. 最终视频中可以真实展示的能力

1. **真实运镜视频与多分支素材的端到端合成**：
   从单条实拍移动视频（Pexels 1280×720，24 fps）输入，到导出 3 个真正可播放的独立 H.264/MP4 视频文件。
2. **亚像素级几何透视贴合**：
   SIFT 特征锚点匹配 + RANSAC 单应矩阵解算，即便镜头出现平移与视角倾斜，替换后的招牌依然无漂移贴合，中位重投影残差仅 0.21 像素。
3. **三路同步输出的一致性保证**：
   三个分支成片共用同一套几何追踪帧序列，保证完全一致的时长、帧数、帧率及动态变形。
4. **端到端 Web 交互体验**：
   支持直观的文件拖拽上传、首帧招牌四角线框可视化渲染、实时性能指标仪表盘、三路视频同时循环预览播放与成片下载。

---

## 7. 已知限制与边界说明（诚实披露）

1. **实时网关网络依赖**：
   GLM 视觉定位接口受限于特定网络白名单/网关配置，在外部测试环境下可能产生 403，录制 Demo 须依托本地手动兜底模式演示，不可依赖现场实时云端调用。
2. **前景物体遮挡（Occlusion）**：
   在 9.7 秒完整视频的第 155 至 161 帧（约占 3.0%），因行人前景穿越招牌，系统对高置信度局部残差保留原始前景像素，跟踪失效帧继续保留原画。长测摘要记录了 20 个遮挡提示，系统仍未加入通用动态遮挡分割层（如 Segment Anything）。录屏演示提供 6.0 秒干净运镜版本（`demo_cut.mp4`），完整片段作为压力测试公开披露。
3. **处理延迟**：
   全分辨率（720p）逐帧 SIFT 提取与透视变换在 CPU 上处理完整 233 帧约需数十秒，更适合单次渲染等待与异步下载，不适合现场实时毫秒级预览。
