"""Infer masks using the exact submitted FP32 TTA geometry and aggregation."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_runtime import load_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['Native', 'TTA4', 'TTA12'], default='TTA12')
    parser.add_argument('--threshold', type=float, default=.46)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0, help='Optional short interface check')
    args = parser.parse_args()
    import cv2
    import mmcv
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    model = load_student(args.checkpoint, args.device)
    from astra.tta import grouped_bundle
    from astra import image_ops as direct
    paths = sorted(p for p in args.images.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError('No images found')
    target = args.output / 'result'
    if args.output.exists():
        raise FileExistsError('Use a new output directory: ' + str(args.output))
    target.mkdir(parents=True)
    written = []
    with torch.inference_mode():
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable image: ' + str(path))
            h, w = image.shape[:2]
            if args.mode == 'TTA12':
                probability = grouped_bundle(model, image)['TTA12']
            else:
                raw = direct.normalize_image(image, model)
                flips = ['none'] if args.mode == 'Native' else ['none', 'horizontal', 'vertical', 'diagonal']
                views = [direct.pad_to_divisor(direct.flip_tensor(raw, flip), 32)[0] for flip in flips]
                outputs = model.encode_decode(torch.stack(views))
                total = torch.zeros((h, w), device=outputs.device, dtype=torch.float32)
                for i, flip in enumerate(flips):
                    logits = direct.flip_tensor(outputs[i:i+1, :, :h, :w], flip)
                    total += (logits[0, 1]-logits[0, 0]).sigmoid() if i == 0 else logits.softmax(1)[0, 1]
                probability = (total / len(flips)).cpu().numpy()
            if probability.shape != (h, w) or not np.isfinite(probability).all():
                raise FloatingPointError('Invalid foreground probabilities')
            relative = path.relative_to(args.images).with_suffix('.png')
            result = target / relative
            result.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((probability > np.float32(args.threshold)).astype(np.uint8)).save(result)
            written.append(result)
            if (index+1) % 50 == 0:
                print('Processed %d/%d' % (index+1, len(paths)), flush=True)
    with zipfile.ZipFile(args.output/'result.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in written:
            archive.write(path, 'result/'+path.relative_to(target).as_posix())
    (args.output/'inference.json').write_text(json.dumps(dict(images=len(written), mode=args.mode,
        threshold=args.threshold, precision='FP32', teacher_at_inference=False,
        checkpoint=str(args.checkpoint), mask_values=[0, 1]), indent=2)+'\n')
    print(args.output/'result.zip')


if __name__ == '__main__':
    main()
