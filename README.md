# Astra：无人机路面裂缝轻量分割

**2026 VLP 挑战赛参赛作品**｜赛道D：UAV-Crack｜方法名与用户名：Astra

基于 EfficientViT-B2 和原生分割头，加入跨尺度细节对齐、方向聚合与一致性门控。训练采用 TransNeXt-B＋UPerNet 教师，结合门控细节重建和熵感知蒸馏；部署仅保留单个学生模型。

| 官网 mIoU | Crack F1 | Crack IoU | Precision | Recall | aAcc |
|---:|---:|---:|---:|---:|---:|
| 82.53 | 80.57 | 67.46 | 80.28 | 80.86 | 97.71 |

部署参数 **15.311M**。Native纯前向 **4.87 ms**（RTX4090、FP32、batch1、384×672 GPU输入）。官网结果采用全量1200张训练、TTA12、阈值0.46。

## 安装

Linux、Python3.8、PyTorch2.0.1、MMCV2.1.0、MMEngine0.10.7。入口自动加载仓库内的MMSegmentation1.2.2必要源码。

```bash
conda env create -f environment.yml
conda activate astra_uavcrack
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
python -m pip install -r requirements.txt
```

## 推理与导出

权重下载：[百度网盘](https://pan.baidu.com/s/1i2ovHhx6NyrPg1xu20Su6w?pwd=khxk)，提取码：`khxk`。解压后将权重放入 `weights/`。数据从赛事官方获取。以下命令在仓库根目录执行：

```bash
DATA_ROOT=/path/to/UAV-Crack-dataset
python tools/infer.py --checkpoint weights/astra_full1200.pth --images "$DATA_ROOT/leftImg8bit/val" --output outputs/official --mode TTA12 --threshold 0.46
python tools/export_onnx.py --checkpoint weights/astra_full1200.pth --output weights/astra_native_384x672.onnx
```

预测输出为 `outputs/official/result.zip`，包含原尺寸、单通道、像素值0/1的PNG。TTA12使用0.75/1/1.25三个尺度与四种翻转组合，也支持 `Native`、`TTA4`。

ONNX对应Native单模型：RGB按ImageNet均值和标准差归一化，底部补零至 `[1,3,384,672]`，输入为FP32；输出两类logits，裁剪回原图后取裂缝概率。

## 训练

准备教师权重 `teacher_transnext_b_upernet.pth` 和预训练权重 `efficientvit_seg_b2_ade20k.pt`，放入 `weights/`。EA-KD作者核心单独获取，使用前阅读 [第三方来源与许可](THIRD_PARTY_NOTICES.md)。

```bash
python tools/fetch_eakd.py
python tools/cache_teacher.py --checkpoint weights/teacher_transnext_b_upernet.pth --data-root "$DATA_ROOT" --split splits/public_full1200.txt --output teacher_cache1200
python tools/train.py --protocol full1200 --data-root "$DATA_ROOT" --weights-dir weights --teacher-cache teacher_cache1200 --work-dir work_dirs/astra1200
```

全量训练采用batch8×15,334步。900/300验证使用 `--protocol train900`，并以 `splits/frame_overlap_seed3407_train.txt` 生成对应教师缓存。具体配置见 [configs/](configs/)。

## 许可

自编代码采用 [Apache-2.0](LICENSE)；第三方实现保留原许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
