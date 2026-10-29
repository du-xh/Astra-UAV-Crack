# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import json
import time
from pathlib import Path
import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class AstraOptimizerUpdateHook(Hook):
    priority = 'LOW'

    def before_train(self, runner):
        self.updates = 0
        self.last_updates = 0
        self.skips = 0
        self.start = time.monotonic()
        self.directory = Path(runner.work_dir)

        def check_step(optimizer, args, kwargs):
            finite = [torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if not finite or not bool(torch.stack(finite).all()):
                raise FloatingPointError('Nonfinite gradients at actual optimizer.step')

        def accepted(optimizer, args, kwargs):
            self.updates += 1
        self.pre = runner.optim_wrapper.optimizer.register_step_pre_hook(check_step)
        self.post = runner.optim_wrapper.optimizer.register_step_post_hook(accepted)
        torch.cuda.reset_peak_memory_stats()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.updates == self.last_updates:
            self.skips += 1
        self.last_updates = self.updates
        iteration = runner.iter + 1
        if iteration % 50 == 0 or iteration == runner.max_iters:
            state = dict(iteration=iteration, successful_optimizer_updates=self.updates, AMP_skipped_iterations=self.skips, loss_scale=runner.optim_wrapper.loss_scaler.get_scale(), training_elapsed_seconds=time.monotonic() - self.start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            target = self.directory / 'optimizer_updates.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(target)
            with (self.directory / 'optimizer_updates.jsonl').open('a') as f:
                f.write(json.dumps(state) + '\n')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {}).update(stage12_variant=runner.model.variant, actual_optimizer_updates=self.updates, AMP_skipped_iterations=self.skips)

    def after_train(self, runner):
        self.pre.remove()
        self.post.remove()
