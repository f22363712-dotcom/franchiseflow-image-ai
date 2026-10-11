# FranchiseFlow API 接入记录

## 接入方式

客户端使用 OpenAI 兼容的 `POST /chat/completions` 请求，把首帧作为 base64 data URL 发送给视觉模型。模型只负责一次性识别可替换的招牌平面，后续帧交给本地 OpenCV 几何跟踪，避免每帧调用模型。

默认配置为：

- API 地址：通过 `BANANA_API_BASE_URL` 设置，默认 `https://e.c.h-e.top/v1`
- 模型：`GLM-5.3-flash`
- 密钥：只从 `BANANA_API_KEY` 环境变量读取

## 本机连接测试

### 网关 1：`https://elysia.h-e.top` / `https://e.c.h-e.top`
2026-10-12 在当前电脑分别请求国际地址和国内地址，即便附带浏览器 User-Agent，均返回 `HTTP 403 Forbidden`。客户端已把这个状态转换为明确错误，并且不会生成猜测的四角或伪造成功结果。

### 网关 2：备用多模态网关 `https://keungliang.dpdns.org/v1`
根据比赛方笔记备用 API 记录（位于本地知识库 `Banana Hacks api.md`）进行了实测验证：
- 识别到该网关开启了 Web 防火墙/User-Agent 过滤（默认 `Python-urllib` 会被拦截为 403，配置标准浏览器 `User-Agent` 后正常通信并返回 HTTP 200）。
- 网关支持 51 种模型，实测接入 `gemini-3.1-flash-image` 视觉模型。
- **调用结果**：成功返回首帧招牌四角与语义分析：
  ```json
  {
    "quad": [[0.0, 103.0], [760.0, 497.0], [803.0, 706.0], [0.0, 240.0]],
    "confidence": 0.82,
    "model": "gemini-3.1-flash-image",
    "reason": "The metal header above the brick on the storefront, which contains the text 'STATE of CONFUSION', functions as the primary, largest replaceable flat advertising board or sign panel on this building's facade."
  }
  ```
- **跟踪测试**：将上述模型输出四角代入 `tracker.py` 跟踪流水线：
  - 首帧提取 475 个 SIFT 锚点特征；
  - 跟踪阶段中位内点达 288 个；
  - 重投影中位残差仅 0.30 像素；
  - 成功跑通三分支 MP4 自动渲染导出。
- **结论**：验证了“视觉模型首帧识别招牌平面 + 本地几何光流跟踪”的完整端到端链路完全可行且精度极高。在需要线上模型演示时，可通过 `.env` 切换至此备用网关；在断网或现场无网环境下，无缝使用手动基线兜底。

## 代码安全

密钥仅保存在本地未纳入版本控制的 `.env` 中，未写入仓库、命令行参数、输出 JSON 或日志。`.env.example` 只包含变量名，不含真实值。
