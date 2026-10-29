# Astra：面向无人机路面裂缝的轻量语义分割

**2026 VLP 挑战赛参赛作品**，赛道D：UAV-Crack。VLP用户名、方法名均为Astra。

## 方法与结果

学生采用EfficientViT-B2与原生分割头，通过32通道支路进行跨尺度细节对齐、方向聚合和一致性门控。训练使用冻结的TransNeXt-B＋UPerNet教师，结合门控细节重建和熵感知输出蒸馏。部署保留一个学生，移除教师及重建组件，并折叠RepDW分支。

|评估|mIoU|Crack F1|Crack IoU|Precision|Recall|aAcc|
|---|---:|---:|---:|---:|---:|---:|
|官网，2026-10-29截图|82.53|80.57|67.46|80.28|80.86|97.71|
|公开900/300划分，TTA12校准|81.5289|79.3707|—|79.7514|78.9935|—|

官网提交采用全量1200张训练的学生、FP32 TTA12、固定阈值0.46。公开验证结果来自独立train900学生，不使用全量权重回测这300张图。

部署参数15,310,826。RTX4090、FP32、batch1、384×672已归一化GPU输入，纯前向4.8683ms；同协议教师206.8403ms。该结果是桌面GPU前向测速。TTA12的RAM图像到CPU概率耗时39.2485ms，不含阈值搜索、磁盘写入。

## 环境

Linux、Python3.8.20、PyTorch2.0.1+cu118、MMCV2.1.0、MMEngine0.10.7、MMSegmentation1.2.2。
仓库保留最终方法需要的MMSegmentation源码子集（`vendor/mmseg`）与自定义模块（`astra/`）；入口自动加载该子集。

```bash
conda env create -f environment.yml
conda activate astra_uavcrack
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install -r requirements.txt
```

`requirements.txt`固定了兼容Python3.8/NumPy1.24的OpenCV4.8版本；本次本地检查使用已有环境，已记录实际版本，尚未重新安装全新conda环境。

## 权重与数据

权重由网盘单独提供，目录说明见网盘包的`文件说明.txt`。下载后保留为`weights/`。维护者上传后将下载链接、提取码写入`download_links.json`，共享有效期至少90天。

数据请从比赛官方渠道获得，不包含在仓库或模型包中。数据根目录应含`leftImg8bit/train`、`gtFine/train`、`leftImg8bit/val`。split文件只有图像标识，不含图像或标签。

## 复现官网推理

在仓库根目录执行，`--images`指向官方300张无标签评测图像：

```bash
python tools/infer.py --checkpoint weights/astra_full1200.pth --images /path/to/UAV-Crack-dataset/leftImg8bit/val --output outputs/official --mode TTA12 --threshold 0.46
```

输出`outputs/official/result.zip`。内部为`result/<原图文件名>.png`，保持原图尺寸，单通道像素值0/1。评测只上传300张预测mask，不附带日志或概率文件。

TTA12为0.75/1/1.25三个尺度×原图、水平、垂直及双向翻转；每个尺度的4个视图合批，先还原几何，再平均前景概率。`--mode Native`或`TTA4`用于速度取舍；0.46是TTA12配置的冻结阈值。

## ONNX

网盘提供已导出的`astra_native_384x672.onnx`。转换命令：

```bash
python tools/export_onnx.py --checkpoint weights/astra_full1200.pth --output weights/astra_native_384x672.onnx
```

该ONNX为单张Native学生：输入`[1,3,384,672]`、FP32、RGB、ImageNet均值/标准差归一化，归一化后以0在底部补6行；输出两类logits。还原378×672后softmax取裂缝概率。它不封装十二视图及PNG写入；官网结果由上面的PyTorch TTA12入口复现。

## 复现蒸馏训练

1. 准备`teacher_transnext_b_upernet.pth`与`efficientvit_seg_b2_ade20k.pt`，放入`weights/`。
2. 阅读`THIRD_PARTY_NOTICES.md`，获得EA-KD作者核心后执行`python tools/fetch_eakd.py`。该代码未随本仓库重新分发或更改许可。
3. 生成公开训练集的native教师缓存。下面命令耗时较长，请自行执行：

```bash
python tools/cache_teacher.py --checkpoint weights/teacher_transnext_b_upernet.pth --data-root /path/to/UAV-Crack-dataset --split splits/frame_overlap_seed3407_train.txt --output teacher_cache900
python tools/train.py --protocol train900 --data-root /path/to/UAV-Crack-dataset --weights-dir weights --teacher-cache teacher_cache900 --work-dir work_dirs/astra900
```

train900采用batch8×16000步，每500步验证，以公开val300上的`mIoU_TTA12_cal`选优；实际选中11500步。全量训练从相同ADE20K预训练重新初始化，以11500×1200/900向上取整为15334步，batch8，warmup667，PolyLR终点21334，无独立验证。

```bash
python tools/cache_teacher.py --checkpoint weights/teacher_transnext_b_upernet.pth --data-root /path/to/UAV-Crack-dataset --split splits/public_full1200.txt --output teacher_cache1200
python tools/train.py --protocol full1200 --data-root /path/to/UAV-Crack-dataset --weights-dir weights --teacher-cache teacher_cache1200 --work-dir work_dirs/astra1200
```

训练采用AdamW、基础学习率1e-4、weight decay0.01、梯度裁剪1、AMP。图像随机水平/垂直翻转及光度增强；缓存随几何增强同步变换。教师只用于训练，未使用隐藏测试标签。完整训练未在本次打包时重复执行。

## 代码结构与来源

`astra/`只包含最终学生、教师、细节对齐、方向门控、重参数化、蒸馏和TTA所需代码；`vendor/mmseg/`是必要框架子集；`configs/`仅保留部署、900/300训练、1200全量训练和教师四份配置；`tools/`提供推理、训练、教师缓存与ONNX导出入口。数据、权重和输出路径通过命令行指定。

仓库不含历史实验目录、训练日志、旧结果、训练队列、备选架构、数据集和模型权重。组件出处与许可见 `THIRD_PARTY_NOTICES.md`。

自编代码采用Apache-2.0；第三方文件保留原许可证与署名，见`THIRD_PARTY_NOTICES.md`和`licenses/`。EA-KD上游未提供明确许可证，其作者核心单独获取；公开再分发该核心前应取得许可。
