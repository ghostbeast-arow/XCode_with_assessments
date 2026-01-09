"""Data augmentation strategies for object detection.

Implements random cropping, horizontal flipping, scale augmentation,
color jittering, and random erasing with bounding-box updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentConfig:
    crop_min_iou: float = 0.5
    flip_prob: float = 0.5
    scale_range: Tuple[float, float] = (0.5, 1.5)
    brightness_delta: float = 0.15
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_delta: float = 10.0
    erase_area_range: Tuple[float, float] = (0.02, 0.4)
    erase_aspect_range: Tuple[float, float] = (0.3, 3.0)


def random_crop(image: np.ndarray, boxes: np.ndarray, config: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    rng = np.random.default_rng()
    for _ in range(10):
        crop_w = rng.integers(int(0.6 * width), width)
        crop_h = rng.integers(int(0.6 * height), height)
        left = rng.integers(0, width - crop_w + 1)
        top = rng.integers(0, height - crop_h + 1)
        right = left + crop_w
        bottom = top + crop_h
        crop_box = np.array([left, top, right, bottom])
        ious = _iou(crop_box[None, :], boxes)
        if ious.max() < config.crop_min_iou:
            continue
        new_image = image[top:bottom, left:right]
        new_boxes = boxes.copy()
        new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]] - left, 0, crop_w)
        new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]] - top, 0, crop_h)
        return new_image, new_boxes
    return image, boxes


def horizontal_flip(image: np.ndarray, boxes: np.ndarray, config: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    if rng.random() > config.flip_prob:
        return image, boxes
    flipped = image[:, ::-1]
    width = image.shape[1]
    new_boxes = boxes.copy()
    new_boxes[:, 0] = width - boxes[:, 2]
    new_boxes[:, 2] = width - boxes[:, 0]
    return flipped, new_boxes


def scale_augmentation(image: np.ndarray, boxes: np.ndarray, config: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    scale = rng.uniform(*config.scale_range)
    height, width = image.shape[:2]
    new_w = int(width * scale)
    new_h = int(height * scale)
    scaled = _resize_nearest(image, (new_h, new_w))
    new_boxes = boxes.copy() * scale
    return scaled, new_boxes


def color_jitter(image: np.ndarray, config: AugmentConfig) -> np.ndarray:
    rng = np.random.default_rng()
    img = image.astype(float) / 255.0
    img = np.clip(img + rng.uniform(-config.brightness_delta, config.brightness_delta), 0, 1)
    contrast = rng.uniform(*config.contrast_range)
    img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
    if img.shape[2] == 3:
        hsv = _rgb_to_hsv(img)
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*config.saturation_range), 0, 1)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-config.hue_delta, config.hue_delta) / 360.0) % 1.0
        img = _hsv_to_rgb(hsv)
    return (img * 255).astype(image.dtype)


def random_erasing(image: np.ndarray, config: AugmentConfig) -> np.ndarray:
    rng = np.random.default_rng()
    height, width = image.shape[:2]
    area = height * width
    erase_area = rng.uniform(*config.erase_area_range) * area
    aspect = rng.uniform(*config.erase_aspect_range)
    erase_h = int(np.sqrt(erase_area / aspect))
    erase_w = int(np.sqrt(erase_area * aspect))
    if erase_h <= 0 or erase_w <= 0 or erase_h >= height or erase_w >= width:
        return image
    top = rng.integers(0, height - erase_h)
    left = rng.integers(0, width - erase_w)
    erased = image.copy()
    fill = rng.integers(0, 256, size=(erase_h, erase_w, image.shape[2]), dtype=image.dtype)
    erased[top : top + erase_h, left : left + erase_w] = fill
    return erased


def _iou(crop_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_left = np.maximum(crop_box[:, 0], boxes[:, 0])
    inter_top = np.maximum(crop_box[:, 1], boxes[:, 1])
    inter_right = np.minimum(crop_box[:, 2], boxes[:, 2])
    inter_bottom = np.minimum(crop_box[:, 3], boxes[:, 3])
    inter_area = np.maximum(0, inter_right - inter_left) * np.maximum(0, inter_bottom - inter_top)
    crop_area = (crop_box[:, 2] - crop_box[:, 0]) * (crop_box[:, 3] - crop_box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = crop_area + boxes_area - inter_area
    return inter_area / np.clip(union, 1e-6, None)


def _resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    new_h, new_w = size
    height, width = image.shape[:2]
    y_idx = (np.linspace(0, height - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, width - 1, new_w)).astype(int)
    return image[y_idx[:, None], x_idx[None, :]]


def _rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=-1)
    minc = np.min(image, axis=-1)
    v = maxc
    deltac = maxc - minc
    s = np.where(maxc == 0, 0, deltac / maxc)
    rc = (maxc - r) / np.where(deltac == 0, 1, deltac)
    gc = (maxc - g) / np.where(deltac == 0, 1, deltac)
    bc = (maxc - b) / np.where(deltac == 0, 1, deltac)
    h = np.zeros_like(maxc)
    h = np.where((r == maxc) & (deltac != 0), bc - gc, h)
    h = np.where((g == maxc) & (deltac != 0), 2.0 + rc - bc, h)
    h = np.where((b == maxc) & (deltac != 0), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    h, s, v = image[..., 0], image[..., 1], image[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    boxes = np.array([[30, 40, 120, 160], [150, 60, 260, 200]], dtype=float)
    cfg = AugmentConfig()
    cropped_image, cropped_boxes = random_crop(image, boxes, cfg)
    flipped_image, flipped_boxes = horizontal_flip(image, boxes, cfg)
    scaled_image, scaled_boxes = scale_augmentation(image, boxes, cfg)
    jittered_image = color_jitter(image, cfg)
    erased_image = random_erasing(image, cfg)
    print("crop shapes", cropped_image.shape, cropped_boxes.shape)
    print("flip shapes", flipped_image.shape, flipped_boxes.shape)
    print("scale shapes", scaled_image.shape, scaled_boxes.shape)
    print("jitter shape", jittered_image.shape)
    print("erase shape", erased_image.shape)
