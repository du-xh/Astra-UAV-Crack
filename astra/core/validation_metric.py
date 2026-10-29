# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import json
from pathlib import Path
import re
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from astra.core.binary_metrics import binary_metrics

@METRICS.register_module()
class UAVArchitectureMetric(BaseMetric):

    def __init__(self, image_ids_file, **kwargs):
        super().__init__(**kwargs)
        self.expected_ids = sorted(Path(image_ids_file).read_text().splitlines())
        self.pending = {}
        self.output_dir = self.iteration = self.epoch_report = None

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            gt = sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            image = Path(sample['img_path'])
            identifier = image.parent.name + '/' + image.stem
            if identifier in self.pending:
                raise ValueError('Duplicate validation image')
            bundle = sample['uav_probability_bundle']
            if not np.isin(gt, [0, 1, 255]).all():
                raise ValueError('Unexpected public labels')
            if any((v.dtype != np.float32 or v.shape != gt.shape or (not np.isfinite(v).all()) for v in bundle.values())):
                raise ValueError('Invalid probability bundle')
            valid, positive = (gt != 255, gt == 1)
            predictions = {'native': bundle['native_margin'] > 0}
            predictions.update({k: bundle[k] > 0.5 for k in ['TTA4', 'TTA12'] if k in bundle})
            per = {}
            for protocol, pred in predictions.items():
                per[protocol] = dict(image_id=identifier, scale=image.parent.name, frame_id=image.parent.name + '/' + re.sub('_\\d+_z$', '', image.stem), TP=int((valid & positive & pred).sum()), FP=int((valid & ~positive & pred).sum()), FN=int((valid & positive & ~pred).sum()), TN=int((valid & ~positive & ~pred).sum()))
            self.pending[identifier] = bundle
            self.results.append(per)

    def compute_metrics(self, results):
        if sorted(self.pending) != self.expected_ids:
            raise ValueError('Validation split differs')
        protocols = {}
        for key in results[0]:
            rows = sorted([r[key] for r in results], key=lambda r: r['image_id'])
            protocols[key] = dict(protocol=key, threshold=0.5, iteration=self.iteration, images=len(rows), frames=len({r['frame_id'] for r in rows}), global_metrics=binary_metrics(rows), per_image=rows, scales={s: binary_metrics([r for r in rows if r['scale'] == s]) for s in sorted({r['scale'] for r in rows})})
        key = 'TTA12' if 'TTA12' in protocols else 'native'
        self.epoch_report = dict(protocols[key], protocol_reports=protocols, selection_metric='mIoU_TTA12' if key == 'TTA12' else None)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / ('iteration_%d.json' % self.iteration)).write_text(json.dumps(self.epoch_report, indent=2) + '\n')
        return {'mIoU_' + k: v['global_metrics']['mIoU'] for k, v in protocols.items()}
