# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import json
from pathlib import Path
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadUAVTrainTeacherMargin(BaseTransform):

    def __init__(self, cache_dir, image_root):
        self.cache_dir = Path(cache_dir)
        self.image_root = Path(image_root)
        manifest = json.loads((self.cache_dir / 'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['split_role'] != 'public_train_only':
            raise ValueError('Teacher cache is not a completed train-only artifact')
        self.image_ids = frozenset(manifest['image_ids'])
        self.expected_hw = tuple(manifest['native_hw'])

    def transform(self, results):
        image_id = Path(results['img_path']).relative_to(self.image_root).with_suffix('').as_posix()
        if image_id not in self.image_ids:
            raise ValueError('Image is outside the frozen training teacher cache: ' + image_id)
        margin = np.load(self.cache_dir / (image_id + '.npy'), allow_pickle=False)
        if margin.dtype != np.float16 or margin.shape != self.expected_hw:
            raise ValueError('Unexpected teacher margin dtype/shape: ' + image_id)
        if margin.shape != results['gt_seg_map'].shape or not np.isfinite(margin).all():
            raise ValueError('Teacher/GT mismatch or nonfinite teacher: ' + image_id)
        results['teacher_margin'] = margin
        results.setdefault('seg_fields', []).append('teacher_margin')
        return results

@TRANSFORMS.register_module()
class PackUAVSegInputsWithTeacher(PackSegInputs):

    def transform(self, results):
        packed = super().transform(results)
        margin = np.ascontiguousarray(results['teacher_margin'][None, ...])
        packed['data_samples'].set_data(dict(teacher_margin=PixelData(data=to_tensor(margin))))
        return packed
