# FranchiseFlow 30 分钟技术实验

实验日期：2026-10-11

## 实验问题

一条有相机运动的真实门店外景视频，能否在第一帧给定一个招牌平面后，稳定追踪该平面，把三张分店招牌素材透视贴合到每一帧，并导出三个可播放 MP4。

## 输入

- 视频：`spikes/franchiseflow/fixtures/real_storefront.mp4`
- 来源：Pexels 视频 `6159882`，原作者 K，来源与许可记录在 `spikes/franchiseflow/fixtures/source.md`
- 视频规格：1280×720，24 fps，233 帧，9.71 秒
- 三张替换素材：`spikes/franchiseflow/assets/`
- 第一帧四角：`spikes/franchiseflow/quad.json`，顺序为左上、右上、右下、左下

## 运行

```powershell
python spikes/franchiseflow/tracker.py `
  --video spikes/franchiseflow/fixtures/real_storefront.mp4 `
  --quad spikes/franchiseflow/quad.json `
  --signs spikes/franchiseflow/assets/riverside.png spikes/franchiseflow/assets/downtown.png spikes/franchiseflow/assets/northpark.png `
  --out-dir spikes/franchiseflow/outputs2 `
  --illumination
```

## 结果

生成了三个可播放视频：

- `spikes/franchiseflow/outputs2/branch_1_riverside.mp4`
- `spikes/franchiseflow/outputs2/branch_2_downtown.mp4`
- `spikes/franchiseflow/outputs2/branch_3_northpark.mp4`

跟踪日志和指标在 `spikes/franchiseflow/outputs2/tracking.jsonl` 与 `summary.json`。

- 224 帧使用首帧 SIFT 锚点匹配成功
- 1 帧使用 LK 光流兜底成功
- 7 帧失败，集中在第 155 至 161 帧，占非首帧约 3.0%
- 成功帧中位内点数 123
- 重投影误差中位数 0.31 像素，95 分位 1.70 像素
- 三个分支都按同一套跟踪结果导出
- 全分辨率处理约 108 至 218 秒，取决于是否开启低频光照近似；当前速度不适合现场实时生成

## 视觉观察

首段、中段和尾段的招牌能保持透视贴合，镜头移动时没有明显漂移。第 155 至 161 帧出现前景物体遮挡和几何判定失败，脚本按安全策略保留原画，不冻结上一帧，也不把失败伪装成成功。当前合成没有遮挡层处理，因此替换牌会覆盖经过它的前景物体；这是一项必须补到正式版本的能力。

## 实验结论

透视追踪与三分支导出可行，且输出已经是最终 MP4。这个结果支持继续做 FranchiseFlow 的素材替换方向。

实验使用的是传统 OpenCV SIFT、RANSAC 单应性和 LK 光流，第一帧四角由人工提供。它只证明了平面跟踪与合成，不能证明图像 AI 已自动识别招牌，也不能直接作为最终黑客松版本。正式 Demo 至少需要把第一帧招牌定位改成图像 AI 识别，或者明确展示“模型自动找平面，传统几何负责稳定跟踪”的分工。

## 当前止损条件

如果加入自动平面识别后，移动镜头仍出现连续漂移、遮挡后无法恢复，或一次三分支生成仍超过可接受等待时间，停止扩展功能，录制当前可复现结果并把失败帧如实放进演示说明。
