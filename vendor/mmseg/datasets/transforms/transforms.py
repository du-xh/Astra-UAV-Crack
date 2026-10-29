# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import mmcv
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms.base import BaseTransform
from numpy import random
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta: int=32, contrast_range: Sequence[float]=(0.5, 1.5), saturation_range: Sequence[float]=(0.5, 1.5), hue_delta: int=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img: np.ndarray, alpha: int=1, beta: int=0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """
        if random.randint(2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """
        if random.randint(2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        img = results['img']
        img = self.brightness(img)
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_delta={self.brightness_delta}, contrast_range=({self.contrast_lower}, {self.contrast_upper}), saturation_range=({self.saturation_lower}, {self.saturation_upper}), hue_delta={self.hue_delta})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):
    """Flip the image & bbox & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and gt_depth_map.
    There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string: the image will be
      ``direction``ly flipped with probability of ``prob`` .
      E.g., ``prob=0.5``, ``direction='horizontal'``,
      then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string: the image will
      be ``direction[i]``ly flipped with probability of
      ``prob/len(direction)``.
      E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
      then image will be horizontally flipped with probability of 0.25,
      vertically with probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string:
      given ``len(prob) == len(direction)``, the image will
      be ``direction[i]``ly flipped with probability of ``prob[i]``.
      E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
      'vertical']``, then image will be horizontally flipped with
      probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_depth_map (optional)

    Added Keys:

    - flip
    - flip_direction
    - swap_seg_labels (optional)

    Args:
        prob (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction'])
        img_shape = results['img'].shape[:2]
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'], img_shape, results['flip_direction'])
        for key in results.get('seg_fields', []):
            if results.get(key, None) is not None:
                results[key] = self._flip_seg_map(results[key], direction=results['flip_direction']).copy()
                results['swap_seg_labels'] = self.swap_seg_labels
