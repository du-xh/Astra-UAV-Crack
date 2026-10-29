# 第三方来源与许可

Apache-2.0仅覆盖本项目有权授权的代码；以下第三方文件保留原作者版权。

|来源|用于本方法的内容|原仓库|许可|
|---|---|---|---|
|MMSegmentation|框架、UPerNet及自定义接入|https://github.com/open-mmlab/mmsegmentation|Apache-2.0|
|EfficientViT|学生骨干、原生分割头、ADE20K预训练加载|https://github.com/mit-han-lab/efficientvit|Apache-2.0|
|TransNeXt|教师骨干、CGLU|https://github.com/DaiShiResearch/TransNeXt|Apache-2.0|
|FreqFusion|局部相似性引导重采样|https://github.com/Linwei-Chen/FreqFusion|Apache-2.0|
|CrackLite|方向聚合算子|https://github.com/sisisichen/CrackLite|MIT|
|PIDNet|跨层亲和度计算|https://github.com/XuJiacong/PIDNet|MIT|
|RepViT|训练期RepDW分支及折叠|https://github.com/THU-MIG/RepViT|Apache-2.0|
|Af-DCD|遮挡特征重建作者核心|https://github.com/OSVAI/Af-DCD|Apache-2.0|
|EA-KD|熵权重与输出KD作者核心|https://github.com/cpsu00/EA-KD|上游未声明，核心不随公开包分发|

改进模块沿用上述已克隆、提取的作者实现；通道、插入位置、监督路由、有效像素处理及均值归一化为项目适配。

EA-KD固定版本为`6fb4c0657d0af422e446a508f86705455f4c4a4b`。`tools/fetch_eakd.py`从该版本提取原样函数，输出至被git忽略的`external/`，不将作者代码标为Apache-2.0。推理入口和发布的学生权重无需该训练依赖。


预训练学生权重来自作者公布的 `han-cai/efficientvit-seg` ADE20K分割权重。

`vendor/mmseg` 是MMSegmentation 1.2.2的必要子集，仅注册本方法使用的组件；`astra/vendor/efficientvit` 保留作者骨干和分割头。`astra/core` 收纳最终方法算子及其训练继承关系，已删除未启用的备选架构和实验调度器。
