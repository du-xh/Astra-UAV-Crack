"""Export the teacher-free native student; check ONNX Runtime parity on CPU.

Input is RGB normalized by ImageNet mean/std and padded to384x672. Output is
two-class logits on that padded grid. Crop to378x672 before binary readout.
The official submission uses PyTorch TTA12; this file is its native deployment
conversion, not an ONNX implementation of the entire twelve-view pipeline.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_runtime import load_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    model = load_student(args.checkpoint, 'cpu')
    class Logits(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, image):
            return self.network.encode_decode(image)
    wrapper = Logits(model).eval()
    torch.manual_seed(3407)
    example = torch.randn(1, 3, 384, 672)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = wrapper(example).numpy()
        torch.onnx.export(wrapper, example, str(args.output), opset_version=args.opset,
            input_names=['normalized_rgb'], output_names=['logits'],
            do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(args.output)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'normalized_rgb': example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-3)
    probability = lambda x: 1/(1+np.exp(np.clip(x[:, 0]-x[:, 1], -80, 80)))
    metrics = dict(status='passed', input_shape=list(example.shape), output_shape=list(actual.shape),
        max_absolute_logit_error=float(np.max(np.abs(actual-expected))),
        max_probability_error=float(np.max(np.abs(probability(actual)-probability(expected)))),
        binary_agreement_at_046=float(np.mean((probability(actual)>.46)==(probability(expected)>.46))),
        opset=args.opset, scope='Native padded384x672 student logits; FP32; synthetic input parity check')
    args.output.with_suffix('.verification.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
