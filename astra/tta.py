import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from astra import image_ops as direct
FLIPS = ['none', 'horizontal', 'vertical', 'diagonal']

@torch.inference_mode()
def grouped_bundle(model, image):
    h, w = image.shape[:2]
    device = next(model.parameters()).device
    total = torch.zeros((h, w), device=device, dtype=torch.float32)
    total4 = torch.zeros_like(total)
    margin = None
    with torch.cuda.amp.autocast(enabled=False):
        for scale in [0.75, 1.0, 1.25]:
            normalized = direct.normalize_image(mmcv.imrescale(image, scale, interpolation='bilinear'), model)
            padded = [direct.pad_to_divisor(direct.flip_tensor(normalized, flip), 32)[0] for flip in FLIPS]
            outputs = model.encode_decode(torch.stack(padded))
            sh, sw = normalized.shape[-2:]
            for index, flip in enumerate(FLIPS):
                logits = direct.flip_tensor(outputs[index:index + 1, :, :sh, :sw], flip)
                logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
                if scale == 1.0 and flip == 'none':
                    delta = logits[0, 1] - logits[0, 0]
                    margin = delta.float().cpu().numpy()
                    probability = delta.sigmoid()
                else:
                    probability = logits.softmax(1)[0, 1]
                total.add_(probability)
                if scale == 1.0:
                    total4.add_(probability)
    bundle = dict(native_margin=margin, TTA4=(total4 / 4).cpu().numpy(), TTA12=(total / 12).cpu().numpy())
    if any((value.shape != (h, w) or value.dtype != np.float32 or (not np.isfinite(value).all()) for value in bundle.values())):
        raise FloatingPointError('Invalid validation bundle')
    return bundle

@torch.inference_mode()
def val_step(self, data):
    results = []
    for raw, sample in zip(data['inputs'], data['data_samples']):
        if raw.dtype != torch.uint8 or raw.shape[0] != 3:
            raise ValueError('Expected packed raw BGR images')
        image = np.ascontiguousarray(raw.permute(1, 2, 0).cpu().numpy())
        bundle = grouped_bundle(self, image)
        output = sample.clone()
        output.set_data(dict(pred_sem_seg=PixelData(data=torch.from_numpy(bundle['TTA12'] > 0.5)[None].long()), uav_probability_bundle=bundle))
        results.append(output)
    return results
