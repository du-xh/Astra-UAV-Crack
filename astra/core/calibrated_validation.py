# Copyright: see THIRD_PARTY_NOTICES.md.
import json
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from mmengine.hooks import Hook
from mmseg.registry import METRICS, HOOKS
from astra.core.validation_metric import UAVArchitectureMetric

def metric_from_counts(tp, fp, fn, tn):
    crack = tp / np.maximum(tp + fp + fn, 1)
    background = tn / np.maximum(tn + fp + fn, 1)
    return 50 * (crack + background)

@METRICS.register_module()
class AstraCalibrationMetric(UAVArchitectureMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2).astype(np.float32)
        self.diagnostics = {}

    def process(self, data_batch, data_samples):
        super().process(data_batch, data_samples)
        for sample in data_samples:
            gt = sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            valid = gt != 255
            foreground = gt[valid] == 1
            bundle = sample['uav_probability_bundle']
            native = torch.from_numpy(bundle['native_margin']).sigmoid().numpy()
            probabilities = dict(native=native, **{key: bundle[key] for key in ['TTA4', 'TTA12']})
            for key, probability in probabilities.items():
                values = probability[valid]
                stats = self.diagnostics.setdefault(key, dict(positive=np.zeros(len(self.thresholds) + 1, dtype=np.int64), negative=np.zeros(len(self.thresholds) + 1, dtype=np.int64), foreground_sum=0.0, background_sum=0.0, minimum=1.0, maximum=0.0))
                bins = np.searchsorted(self.thresholds, values, side='left')
                stats['positive'] += np.bincount(bins[foreground], minlength=len(self.thresholds) + 1)
                stats['negative'] += np.bincount(bins[~foreground], minlength=len(self.thresholds) + 1)
                stats['foreground_sum'] += float(values[foreground].sum(dtype=np.float64))
                stats['background_sum'] += float(values[~foreground].sum(dtype=np.float64))
                stats['minimum'] = min(stats['minimum'], float(values.min()))
                stats['maximum'] = max(stats['maximum'], float(values.max()))

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        diagnostics = {}
        for key, stats in self.diagnostics.items():
            positive, negative = (int(stats['positive'].sum()), int(stats['negative'].sum()))
            tp = positive - np.cumsum(stats['positive'])[:-1]
            fp = negative - np.cumsum(stats['negative'])[:-1]
            fn, tn = (positive - tp, negative - fp)
            scores = metric_from_counts(tp, fp, fn, tn)
            index = int(scores.argmax())
            diagnostics[key] = dict(best_threshold=float(self.thresholds[index]), best_mIoU=float(scores[index]), probability_min=stats['minimum'], probability_max=stats['maximum'], foreground_mean=stats['foreground_sum'] / max(1, positive), background_mean=stats['background_sum'] / max(1, negative), background_only_mIoU=50 * negative / max(1, positive + negative), selected_counts=dict(TP=int(tp[index]), FP=int(fp[index]), FN=int(fn[index]), TN=int(tn[index])), grid=[dict(threshold=float(t), mIoU=float(s)) for t, s in zip(self.thresholds, scores)])
            metrics['mIoU_' + key + '_cal'] = float(scores[index])
            metrics['threshold_' + key] = float(self.thresholds[index])
        self.epoch_report.update(probability_diagnostics=diagnostics, selection_metric='mIoU_TTA12_cal', calibration_note='Public val300 grid0.01:0.01:0.99; fixed0.5 metrics also retained; exploratory selection')
        (self.output_dir / ('iteration_%d.json' % self.iteration)).write_text(json.dumps(self.epoch_report, indent=2) + '\n')
        self.diagnostics.clear()
        return metrics

@HOOKS.register_module()
class UAVRepairValidationHook(Hook):
    priority = 'LOWEST'

    def before_train(self, runner):
        self.collapse_count = 0
        model = runner.model
        report = dict(model.pretrained_load_report, checkpoint=runner.cfg.load_from, optimizer_resume=False, actual_batch=runner.cfg.train_dataloader.batch_size)
        (Path(runner.work_dir) / 'pretrained_load_report.json').write_text(json.dumps(report, indent=2) + '\n')

    def _metric(self, runner):
        metric, = [item for item in runner.val_loop.evaluator.metrics if isinstance(item, AstraCalibrationMetric)]
        return metric

    def before_val_epoch(self, runner):
        metric = self._metric(runner)
        if metric.pending:
            raise RuntimeError('Previous validation cache not committed')
        metric.iteration = runner.iter
        metric.output_dir = Path(runner.work_dir) / 'validation'

    def after_val_epoch(self, runner, metrics=None):
        metric = self._metric(runner)
        work = Path(runner.work_dir)
        checkpoint, = work.glob('best_mIoU_TTA12_cal_iter_*.pth')
        iteration = int(re.search('iter_(\\d+)\\.pth$', checkpoint.name).group(1))
        selected = metric.epoch_report['probability_diagnostics']['TTA12']
        if iteration == metric.iteration:
            target, backup = (work / 'best_validation_cache', work / 'best_validation_cache.previous')
            temporary = work / ('best_validation_cache.pending_%d' % iteration)
            if temporary.exists() or backup.exists():
                raise RuntimeError('Incomplete cache transaction')
            temporary.mkdir()
            for identifier in metric.expected_ids:
                path = temporary / (identifier + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(path), **metric.pending[identifier])
            (temporary / 'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), selected_iteration=iteration, selected_threshold=selected['best_threshold'], selected_score=selected['best_mIoU'], images=len(metric.expected_ids), image_ids=metric.expected_ids, FP32=True, serial_TTA12=runner.model.tta_view_batch == 1, hidden_inputs=False, TTA_view_batch=runner.model.tta_view_batch, TTA_forward_calls=12 // runner.model.tta_view_batch, selection='public calibrated mIoU; fixed0.5 recorded separately'), indent=2) + '\n')
            if target.exists():
                target.rename(backup)
            temporary.rename(target)
            if backup.exists():
                shutil.rmtree(backup)
        metric.pending.clear()
        no_skill = selected['best_mIoU'] <= selected['background_only_mIoU'] + 0.5
        self.collapse_count = self.collapse_count + 1 if no_skill else 0
        if self.collapse_count >= 2 and runner.iter >= 2000:
            (work / 'collapse_guard.json').write_text(json.dumps(dict(iteration=runner.iter, reason='Two public validations at background baseline even after calibration', diagnostics=selected), indent=2) + '\n')
            runner.train_loop.stop_training = True
            runner.logger.error('Stopping: two calibrated validations at the background baseline')
