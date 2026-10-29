"""Cache only public-training native teacher margins for aligned augmentation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_runtime import ROOT, setup, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from mmengine.model import revert_sync_batchnorm
    from mmseg.registry import MODELS
    setup()
    teacher = MODELS.build(config('teacher.py').model)
    teacher.load_state_dict(torch.load(str(args.checkpoint), map_location='cpu')['state_dict'], strict=True)
    teacher = revert_sync_batchnorm(teacher).to(args.device).eval().requires_grad_(False)
    from astra import image_ops as direct
    names = [s.strip() for s in args.split.read_text().splitlines() if s.strip()]
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
        for i, name in enumerate(names):
            image = cv2.imread(str(args.data_root/'leftImg8bit/train'/(name+'.jpg')))
            if image is None or image.shape != (378, 672, 3):
                raise ValueError(name)
            raw = direct.normalize_image(image, teacher)
            padded = direct.pad_to_divisor(raw, 32)[0]
            logits = teacher.encode_decode(padded[None], [dict(ori_shape=(378,672),
                img_shape=(384,672), pad_shape=(384,672), padding_size=[0,0,0,6])])
            margin = (logits[0,1,:378,:672]-logits[0,0,:378,:672]).cpu().numpy().astype(np.float16)
            path = args.output/(name+'.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, margin)
            if (i+1)%50 == 0:
                print('Cached %d/%d' % (i+1,len(names)), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(status='complete',
        split_role='public_train_only', image_ids=names, native_hw=[378,672],
        storage='float16 native crack-minus-background logits', teacher_checkpoint=str(args.checkpoint)),indent=2)+'\n')


if __name__ == '__main__':
    main()
