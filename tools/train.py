"""Reproduce the selected training recipe; never reads official hidden labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_runtime import ROOT, setup, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['train900','full1200'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true',
        help='Build the CPU training graph and print the recipe; do not train')
    args = parser.parse_args()
    from mmengine.runner import Runner
    setup()
    cfg = config('astra_train900.py' if args.protocol=='train900' else 'astra_full1200.py')
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.model.pretrained_checkpoint = str(args.weights_dir/'efficientvit_seg_b2_ade20k.pt')
    cfg.model.feature_teacher_checkpoint = str(args.weights_dir/'teacher_transnext_b_upernet.pth')
    cfg.model.feature_teacher_config = str(ROOT/'configs/teacher.py')
    split = 'frame_overlap_seed3407_train.txt' if args.protocol=='train900' else 'public_full1200.txt'
    for role in ['train_dataloader','val_dataloader']:
        loader = cfg.get(role)
        if loader is None:
            continue
        loader.dataset.data_root = str(args.data_root.resolve())
        loader.dataset.ann_file = str(ROOT/'splits'/(
            split if role=='train_dataloader' else 'frame_overlap_seed3407_val.txt'))
        for transform in loader.dataset.pipeline:
            if transform['type']=='LoadUAVTrainTeacherMargin':
                transform['cache_dir']=str(args.teacher_cache.resolve())
                transform['image_root']=str(args.data_root.resolve()/'leftImg8bit/train')
    if cfg.get('val_evaluator'):
        cfg.val_evaluator.image_ids_file=str(ROOT/'splits/frame_overlap_seed3407_val.txt')
    if args.dry_run:
        import json
        from mmseg.registry import MODELS
        model=MODELS.build(cfg.model)
        count=sum(p.numel() for p in model.parameters())
        if count!=20047466:
            raise RuntimeError('Training graph mismatch: '+str(count))
        print(json.dumps(dict(status='passed',protocol=args.protocol,training_parameters=count,
            batch=cfg.train_dataloader.batch_size,iterations=cfg.train_cfg.max_iters,
            validation_enabled=cfg.val_cfg is not None,teacher_constructed=model._feature_teacher is not None,
            split_file=cfg.train_dataloader.dataset.ann_file,training_executed=False),indent=2))
        return
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
