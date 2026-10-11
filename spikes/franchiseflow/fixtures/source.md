# FranchiseFlow 实验素材来源

## 本地素材

- 文件：`real_storefront.mp4`
- 首帧：`real_storefront_first.jpg`
- 处理：从原片截取前 10 秒，缩放至 1280×720，保留原始镜头运动；未把静态照片动画化。
- 生成日期：2026-10-11

## 原始来源

- 页面：https://www.pexels.com/video/a-brick-walled-store-with-a-signage-6159882/
- 直接下载地址：https://videos.pexels.com/video-files/6159882/6159882-uhd_2560_1440_24fps.mp4
- 作者：K（Pexels contributor 页面显示为 K，个人页 https://www.pexels.com/@kelly/）
- 页面描述：Revolving shot of a brick storefront named “State of Confusion”.
- 原片信息：3840×2160，16:9，10 秒，24 fps。
- 许可：Pexels Free License，参见 https://www.pexels.com/license/ 。用于黑客松原型演示；提交时保留来源与作者信息。

## 适合 FranchiseFlow 的画面区域

首帧有一个明显的深灰色矩形门店招牌，位于画面上方偏中，约覆盖 `x=360..1080, y=80..440`（1280×720 坐标）；“STATE OF CONFUSION”字样的主要区域约在 `x=430..950, y=135..400`。招牌平面具有明显的水平透视，镜头沿门店立面移动/旋转，约中段会被前景立柱短暂遮挡，适合测试角点跟踪、透视贴合和遮挡恢复。

## 限制

该素材是公共版权库中的真实门店外景，不代表加盟店实际素材。招牌文字较大但前景立柱会造成遮挡，若自动跟踪不稳定，实验可记录为失败并更换较简单的受控素材。
