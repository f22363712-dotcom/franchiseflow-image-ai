# FranchiseFlow API 接入记录

## 接入方式

客户端使用 OpenAI 兼容的 `POST /chat/completions` 请求，把首帧作为 base64 data URL 发送给视觉模型。模型只负责一次性识别可替换的招牌平面，后续帧交给本地 OpenCV 几何跟踪，避免每帧调用模型。

默认配置为：

- API 地址：通过 `BANANA_API_BASE_URL` 设置，默认 `https://e.c.h-e.top/v1`
- 模型：`GLM-5.3-flash`
- 密钥：只从 `BANANA_API_KEY` 环境变量读取

## 本机连接测试

2026-10-12 在当前电脑分别请求国际地址和国内地址，均返回 HTTP 403。客户端已把这个状态转换为明确错误，并且不会生成猜测的四角或伪造成功结果。

最新实测验证（2026-10-12）：
- 执行 `python spikes/franchiseflow/auto_quad.py --image spikes/franchiseflow/fixtures/real_storefront_first.jpg --out-json spikes/franchiseflow/quad_auto.json --preview spikes/franchiseflow/quad_auto_preview.jpg`
- 接口真实返回：`Vision API error: vision gateway returned HTTP 403; check base URL, key, and gateway access`
- 客户端未产生 `quad_auto.json`，未产生预览图，未在日志或输出中泄漏任何 API Key 或鉴权头部。
- 根据计划止损条件：由于接口返回 HTTP 403，记录真实错误并停止对实时 API 的依赖；不可使用手工标注四角冒充模型在线识别；演示与离线验证明确标注四角为手动初始化兜底（manual engineering fallback）。

这说明当前还不能把网关访问当作已验证能力。可能原因包括网关需要特定来源、访问白名单、不同请求路径或额外协议约定。若后续比赛方提供更正后的网关地址或网络环境，可通过同一接口重新验证。

## 代码安全

密钥没有写入仓库、命令行参数、输出 JSON 或日志。`.env.example` 只包含变量名，不含真实值。
