# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
from pathlib import Path
import torch
import torch.nn as nn
from astra.vendor.efficientvit.seg_model_zoo import create_efficientvit_seg_model

def load_author_core(kind, checkpoint=None, binary=True):
    if kind == 'efficientvit_b2':
        model = create_efficientvit_seg_model('efficientvit-seg-b2-ade20k', pretrained=False)
        classifier = model.head.output_ops[0].op_list[0].conv
        classifier_path = 'head.output_ops.0.op_list.0.conv'
    else:
        raise ValueError('Unsupported student: ' + str(kind))
    report = dict(kind=kind, checkpoint=None if checkpoint is None else str(checkpoint), full_semantic_classes=150, strict_loaded=False, excluded_training_only_keys=[], missing_keys=[], unexpected_keys=[], classifier_reset=False)
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        obj = torch.load(str(path), map_location='cpu', weights_only=False)
        state = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
        state = {k.removeprefix('module.') if hasattr(k, 'removeprefix') else k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        expected = model.state_dict()
        if set(state) != set(expected):
            raise RuntimeError('Author inference keys differ: missing=' + str(sorted(set(expected) - set(state))) + '; unexpected=' + str(sorted(set(state) - set(expected))))
        mismatch = [k for k in state if state[k].shape != expected[k].shape]
        if mismatch:
            raise RuntimeError('Author inference tensor shapes differ: ' + str(mismatch))
        model.load_state_dict(state, strict=True)
        report.update(strict_loaded=True, loaded_tensors=len(state), loaded_parameter_elements=sum((p.numel() for p in model.parameters())))
    if binary:
        replacement = nn.Conv2d(classifier.in_channels, 2, classifier.kernel_size, classifier.stride, classifier.padding, classifier.dilation, classifier.groups, bias=classifier.bias is not None)
        nn.init.normal_(replacement.weight, std=0.01)
        if replacement.bias is not None:
            nn.init.zeros_(replacement.bias)
        if kind == 'efficientvit_b2':
            model.head.output_ops[0].op_list[0].conv = replacement
        else:
            model.decode_head.conv_seg = replacement
            model.decode_head.num_classes = 2
            model.decode_head.out_channels = 2
        report.update(classifier_reset=True, reset_keys=[classifier_path + '.weight', classifier_path + '.bias'], binary_parameters=sum((p.numel() for p in model.parameters())), new_classifier_init='normal std0.01, zero bias')
    model.pretrained_load_report = report
    return model
