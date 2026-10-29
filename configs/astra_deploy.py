custom_imports = {'imports': ['astra.core.assembly'], 'allow_failed_imports': False}

default_scope = 'mmseg'

model = {'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (384, 672),
                       'std': [58.395, 57.12, 57.375],
                       'test_cfg': {'size_divisor': 32},
                       'type': 'SegDataPreProcessor'},
 'initial_checkpoint': None,
 'kd_mode': 'ordinary',
 'kd_tau': 2.0,
 'kd_weight': 0.0,
 'pretrained_checkpoint': None,
 'student_kind': 'efficientvit_b2',
 'test_cfg': {'mode': 'whole'},
 'train_cfg': {},
 'transfer_init_checkpoint': None,
 'transfer_kind': 'cglu32',
 'type': 'AstraDeploymentSegmentor',
 'variant': 'D4-B'}
