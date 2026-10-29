import numpy as np
import torch
import torch.nn.functional as F
FLIP_DIMS = {'none': (), 'horizontal': (-1,), 'vertical': (-2,), 'diagonal': (-2, -1)}

def normalize_image(image, model):
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).to(device=next(model.parameters()).device, dtype=torch.float32)
    preprocessor = model.data_preprocessor
    if preprocessor.channel_conversion:
        tensor = tensor[[2, 1, 0], ...]
    if preprocessor._enable_normalize:
        tensor = (tensor - preprocessor.mean) / preprocessor.std
    return tensor

def flip_tensor(tensor, direction):
    dims = FLIP_DIMS[direction]
    return tensor.flip(dims=dims) if dims else tensor

def pad_to_divisor(tensor, divisor):
    height, width = tensor.shape[-2:]
    pad_height = -height % divisor
    pad_width = -width % divisor
    padded = F.pad(tensor, (0, pad_width, 0, pad_height), value=0.0)
    return (padded, (height, width), (pad_height, pad_width))
