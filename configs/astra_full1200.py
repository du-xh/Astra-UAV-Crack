custom_hooks = [{'priority': 'NORMAL', 'type': 'AstraOptimizerUpdateHook'},
 {'type': 'AstraObjectiveGuard'},
 {'type': 'AstraPrecisionAuditHook'},
 {'type': 'AstraFullFitHook'}]

custom_imports = {'imports': ['astra.core.entropy_distillation', 'astra.core.optimizer_hook', 'astra.core.data'],
 'allow_failed_imports': False}

default_hooks = {'checkpoint': {'by_epoch': False,
                'interval': -1,
                'max_keep_ckpts': 1,
                'save_best': None,
                'save_last': True,
                'save_optimizer': False,
                'save_param_scheduler': False,
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'log_metric_by_epoch': False, 'type': 'LoggerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'}}

default_scope = 'mmseg'

env_cfg = {'cudnn_benchmark': True,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

launcher = 'none'

load_from = None

log_level = 'INFO'

log_processor = {'by_epoch': False}

model = {'contrast_enabled': False,
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (384, 672),
                       'std': [58.395, 57.12, 57.375],
                       'test_cfg': {'size_divisor': 32},
                       'type': 'SegDataPreProcessor'},
 'feature_contrast_weight': 0.0,
 'feature_mse_weight': 2.0903356466227648e-07,
 'feature_target_mode': 'pooled_target',
 'feature_teacher_checkpoint': 'weights/teacher_transnext_b_upernet.pth',
 'feature_teacher_config': 'configs/teacher.py',
 'initial_checkpoint': None,
 'kd_mode': 'ordinary',
 'kd_tau': 2.0,
 'kd_weight': 1.0,
 'night_objective': 'EAKD',
 'pretrained_checkpoint': 'weights/efficientvit_seg_b2_ade20k.pt',
 'projector_amp': True,
 'scnp_weight': 0.25,
 'semantic_temperature': 2.0,
 'semantic_weight': 0.1,
 'student_kind': 'efficientvit_b2',
 'teacher_feature_batch': 8,
 'test_cfg': {'mode': 'whole'},
 'train_cfg': {},
 'transfer_init_checkpoint': None,
 'transfer_kind': 'cglu32',
 'type': 'AstraDistillationSegmentor',
 'variant': 'D4-B'}

night_fullfit = {'arm': 'EAKD',
 'source_selected_iteration': 11500,
 'training_images': 1200,
 'warmup': 667,
 'PolyLR_end': 21334}

optim_wrapper = {'accumulative_counts': 1,
 'clip_grad': {'max_norm': 1.0, 'norm_type': 2},
 'loss_scale': {'growth_interval': 2000, 'init_scale': 512.0},
 'optimizer': {'betas': (0.9, 0.999), 'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.01},
 'paramwise_cfg': {'bias_decay_mult': 0.0,
                   'custom_keys': {'context': {'lr_mult': 10.0},
                                   'detail.residual_logits': {'lr_mult': 10.0},
                                   'feature_distill': {'lr_mult': 10.0},
                                   'fusion': {'lr_mult': 10.0},
                                   'network.decode_head.conv_seg': {'lr_mult': 10.0},
                                   'network.head.output_ops.0.op_list.0.conv': {'lr_mult': 10.0},
                                   'semantic_control': {'lr_mult': 10.0},
                                   'transfer': {'lr_mult': 10.0}},
                   'norm_decay_mult': 0.0},
 'type': 'AmpOptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 667, 'start_factor': 0.01, 'type': 'LinearLR'},
 {'begin': 667, 'by_epoch': False, 'end': 21334, 'eta_min': 0.0, 'power': 1.0, 'type': 'PolyLR'}]

randomness = {'deterministic': False, 'seed': 3407}

resume = False

test_cfg = None

test_dataloader = None

test_evaluator = None

train_cfg = {'max_iters': 15334, 'type': 'IterBasedTrainLoop', 'val_begin': 15335, 'val_interval': 15335}

train_dataloader = {'batch_size': 8,
 'dataset': {'ann_file': 'splits/public_full1200.txt',
             'data_prefix': {'img_path': 'leftImg8bit/train', 'seg_map_path': 'gtFine/train'},
             'data_root': 'data/',
             'img_suffix': '.jpg',
             'metainfo': {'classes': ('normal', 'crack'), 'palette': [[0, 0, 0], [255, 255, 255]]},
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'reduce_zero_label': False, 'type': 'LoadAnnotations'},
                          {'cache_dir': 'teacher_cache',
                           'image_root': 'data/leftImg8bit/train',
                           'type': 'LoadUAVTrainTeacherMargin'},
                          {'direction': 'horizontal', 'prob': 0.5, 'type': 'RandomFlip'},
                          {'direction': 'vertical', 'prob': 0.5, 'type': 'RandomFlip'},
                          {'type': 'PhotoMetricDistortion'},
                          {'meta_keys': ('img_path',
                                         'seg_map_path',
                                         'ori_shape',
                                         'img_shape',
                                         'pad_shape',
                                         'scale_factor',
                                         'flip',
                                         'flip_direction',
                                         'reduce_zero_label'),
                           'type': 'PackUAVSegInputsWithTeacher'}],
             'reduce_zero_label': False,
             'seg_map_suffix': '.png',
             'test_mode': False,
             'type': 'BaseSegDataset'},
 'drop_last': True,
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

val_cfg = None

val_dataloader = None

val_evaluator = None

visualizer = None

work_dir = 'work_dirs/astra'
