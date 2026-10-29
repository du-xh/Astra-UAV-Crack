"""Check the standalone final student using supplied weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_runtime import load_student, ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    import torch
    model = load_student(args.checkpoint, 'cpu')
    with torch.inference_mode():
        output = model.encode_decode(torch.zeros(1, 3, 384, 672))
    if output.shape != (1, 2, 384, 672) or not torch.isfinite(output).all():
        raise RuntimeError('Invalid student output')
    import mmseg
    if not str(Path(mmseg.__file__).resolve()).startswith(str(ROOT / 'vendor')):
        raise RuntimeError('The bundled MMSeg subset was not loaded')
    print(json.dumps(dict(status='passed', deployment_parameters=sum(p.numel() for p in model.parameters()),
                         output_shape=list(output.shape)), indent=2))

if __name__ == '__main__':
    main()
