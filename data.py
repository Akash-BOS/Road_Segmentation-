"""LCMS crack dataset and lightweight transforms."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

LOGGER = logging.getLogger("lcms_unetpp.data")

try:
    cv2.setNumThreads(0)
except cv2.error:
    pass

IGNORE_INDEX = 255

COLOR_MAP: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 0, 255): 2,
    (0, 255, 0): 3,
    (139, 69, 19): 4,
    (255, 165, 0): 5,
}


class LCMSCrackDataset(Dataset):
    """Dataset for multi-class semantic segmentation.

    Expected layout::

        root/TRAIN/IMAGES
        root/TRAIN/MASKS
        root/VAL/IMAGES
        root/VAL/MASKS
        root/TEST/IMAGES
        root/TEST/MASKS

    Args:
        root: Dataset root.
        split: ``TRAIN``, ``VAL``, or ``TEST``.
        transform: Callable receiving ``image, mask`` and returning tensors.
        mask_mode: ``color`` for RGB masks, ``index`` for class-id masks, or ``binary``.
        color_map: RGB-to-class mapping used when ``mask_mode='color'``.

    Images may be grayscale, RGB, or multi-channel arrays. Masks are returned as
    integer class IDs with shape ``[H, W]``.
    """

    def __init__(
        self,
        root: str,
        split: str = "TRAIN",
        transform: Callable | None = None,
        mask_mode: str = "color",
        color_map: Dict[Tuple[int, int, int], int] | None = None,
    ) -> None:
        self.root = Path(root) / split.upper()
        self.images_dir = self.root / "IMAGES"
        self.masks_dir = self.root / "MASKS"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Missing mask folder: {self.masks_dir}")
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in valid_exts])
        self.mask_paths = [self.masks_dir / f"{p.stem}.png" for p in self.image_paths]
        self.transform = transform
        self.mask_mode = mask_mode.lower()
        self.color_map = color_map or COLOR_MAP
        missing_masks = [p for p in self.mask_paths if not p.exists()]
        if missing_masks:
            preview = ", ".join(str(p) for p in missing_masks[:5])
            LOGGER.error("missing_masks count=%d first=%s", len(missing_masks), preview)
        LOGGER.info(
            "dataset split=%s root=%s images=%d masks_dir=%s mask_mode=%s",
            split.upper(),
            self.root,
            len(self.image_paths),
            self.masks_dir,
            self.mask_mode,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._read_image(self.image_paths[index])
        mask = self._read_mask(self.mask_paths[index])
        mask = self._prepare_mask(mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask.long()

    def _read_image(self, image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as image_img:
                image = np.array(image_img.convert("RGB"), dtype=np.uint8, copy=True)
                return np.ascontiguousarray(image)
        except FileNotFoundError:
            raise FileNotFoundError(image_path) from None
        except Exception:
            LOGGER.exception("failed_to_read_image path=%s", image_path)
            raise

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        try:
            with Image.open(mask_path) as mask_img:
                if self.mask_mode == "color":
                    return np.array(mask_img.convert("RGB"), dtype=np.uint8, copy=True)
                if self.mask_mode in {"index", "binary"}:
                    return np.array(mask_img.convert("L"), dtype=np.uint8, copy=True)
                return np.array(mask_img, dtype=np.uint8, copy=True)
        except FileNotFoundError:
            raise FileNotFoundError(mask_path) from None
        except Exception:
            LOGGER.exception("failed_to_read_mask path=%s", mask_path)
            raise

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, None]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_mode == "binary":
            if mask.ndim == 3:
                return (mask.sum(axis=2) > 0).astype(np.uint8)
            return (mask > 0).astype(np.uint8)
        if self.mask_mode == "index":
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return mask.astype(np.uint8)
        if self.mask_mode == "color":
            if mask.ndim == 2:
                return mask.astype(np.uint8)
            class_map = np.full(mask.shape[:2], IGNORE_INDEX, dtype=np.uint8)
            for color, class_id in self.color_map.items():
                r, g, b = color
                matches = (mask[:, :, 0] == r) & (mask[:, :, 1] == g) & (mask[:, :, 2] == b)
                class_map[matches] = class_id
            return class_map
        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return _cat_list(images, 0), _cat_list(targets, IGNORE_INDEX)


class LCMSCrackPatchDataset(LCMSCrackDataset):
    """Virtual patch dataset with class-balanced anomaly sampling.

    The dataset keeps the original files on disk and samples many online crops per
    epoch. Each sample chooses either an anomaly-centered crop or a mostly
    background crop. Rare classes can be oversampled by increasing their
    ``class_sampling_weights``.
    """

    def __init__(
        self,
        root: str,
        split: str = "TRAIN",
        transform: Callable | None = None,
        mask_mode: str = "color",
        color_map: Dict[Tuple[int, int, int], int] | None = None,
        samples_per_image: int = 4,
        anomaly_ratio: float = 0.7,
        random_crop_ratio: float = 0.1,
        class_sampling_weights: Dict[int, float] | None = None,
        min_target_pixels: int = 24,
        target_crop_attempts: int = 32,
        foreground_extension_ratio: float = 0.0,
    ) -> None:
        super().__init__(root, split, transform, mask_mode, color_map)
        self.samples_per_image = max(1, samples_per_image)
        self.anomaly_ratio = min(max(anomaly_ratio, 0.0), 1.0)
        self.random_crop_ratio = min(max(random_crop_ratio, 0.0), 1.0)
        self.class_sampling_weights = class_sampling_weights or {1: 2.0, 2: 6.0, 3: 5.0, 4: 2.0, 5: 2.0}
        self.min_target_pixels = max(1, min_target_pixels)
        self.target_crop_attempts = max(1, target_crop_attempts)
        self.foreground_extension_ratio = max(0.0, foreground_extension_ratio)
        self.class_to_indices = self._scan_mask_presence()
        self.foreground_classes = [class_id for class_id, indices in self.class_to_indices.items() if indices]
        LOGGER.info(
            "patch_dataset samples_per_image=%d virtual_samples=%d anomaly_ratio=%.2f random_crop_ratio=%.2f class_counts=%s weights=%s",
            self.samples_per_image,
            len(self),
            self.anomaly_ratio,
            self.random_crop_ratio,
            {class_id: len(indices) for class_id, indices in self.class_to_indices.items()},
            self.class_sampling_weights,
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_image

    def _scan_mask_presence(self) -> Dict[int, list[int]]:
        class_to_indices: Dict[int, list[int]] = {class_id: [] for class_id in range(1, 6)}
        for index, mask_path in enumerate(self.mask_paths):
            mask = self._prepare_mask(self._read_mask(mask_path))
            present = set(np.unique(mask).tolist())
            for class_id in class_to_indices:
                if class_id in present:
                    class_to_indices[class_id].append(index)
            if index % 100 == 0:
                LOGGER.info("patch_dataset scanned mask=%d/%d path=%s", index, len(self.mask_paths), mask_path.name)
        return class_to_indices

    def _choose_target_class(self) -> int | None:
        available = [class_id for class_id in self.foreground_classes if self.class_to_indices[class_id]]
        if not available:
            return None
        weights = [self.class_sampling_weights.get(class_id, 1.0) for class_id in available]
        return random.choices(available, weights=weights, k=1)[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        force_random_crop = random.random() < self.random_crop_ratio
        force_background = (not force_random_crop) and random.random() >= self.anomaly_ratio
        target_class = None if force_background or force_random_crop else self._choose_target_class()
        if target_class is not None:
            image_index = random.choice(self.class_to_indices[target_class])
        else:
            image_index = random.randrange(len(self.image_paths))

        image = self._read_image(self.image_paths[image_index])
        mask = self._read_mask(self.mask_paths[image_index])
        mask = self._prepare_mask(mask)
        if self.transform is not None:
            try:
                image, mask = self.transform(
                    image,
                    mask,
                    target_class=target_class,
                    force_background=force_background,
                    force_random_crop=force_random_crop,
                    min_target_pixels=self.min_target_pixels,
                    target_crop_attempts=self.target_crop_attempts,
                    foreground_extension_ratio=self.foreground_extension_ratio,
                )
            except TypeError:
                image, mask = self.transform(image, mask)
        return image, mask.long()


class MixedFullPatchDataset(Dataset):
    """Mix full-image samples into an online patch dataset."""

    def __init__(
        self,
        patch_dataset: Dataset,
        full_dataset: Dataset,
        full_image_ratio: float = 0.1,
    ) -> None:
        self.patch_dataset = patch_dataset
        self.full_dataset = full_dataset
        self.full_image_ratio = min(max(full_image_ratio, 0.0), 1.0)
        LOGGER.info(
            "mixed_dataset patch_samples=%d full_samples=%d full_image_ratio=%.2f",
            len(self.patch_dataset),
            len(self.full_dataset),
            self.full_image_ratio,
        )

    def __len__(self) -> int:
        return len(self.patch_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.full_image_ratio > 0 and random.random() < self.full_image_ratio:
            full_index = random.randrange(len(self.full_dataset))
            return self.full_dataset[full_index]
        return self.patch_dataset[index]


class TrainTransform:
    """Basic crack-preserving augmentation and normalization."""

    def __init__(self, height: int, width: int, mean: float = 0.5, std: float = 0.25) -> None:
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = _resize(image, mask, self.height, self.width)
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.2:
            image = np.ascontiguousarray(image[::-1])
            mask = np.ascontiguousarray(mask[::-1])
        return _to_tensor(image, mask, self.mean, self.std)


class BiasedPatchTrainTransform:
    """Anomaly-biased patch crop with mild geometric and photometric augmentation."""

    def __init__(
        self,
        height: int,
        width: int,
        mean: float = 0.5,
        std: float = 0.25,
        max_rotation: float = 12.0,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        perspective: float = 0.025,
        background_max_foreground_fraction: float = 0.002,
    ) -> None:
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.max_rotation = max_rotation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.perspective = perspective
        self.background_max_foreground_fraction = background_max_foreground_fraction

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        target_class: int | None = None,
        force_background: bool = False,
        force_random_crop: bool = False,
        min_target_pixels: int = 24,
        target_crop_attempts: int = 32,
        foreground_extension_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._crop_patch(
            image,
            mask,
            target_class,
            force_background,
            force_random_crop,
            min_target_pixels,
            target_crop_attempts,
            foreground_extension_ratio,
        )
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        image, mask = self._geometric(image, mask)
        image = self._photometric(image)
        return _to_tensor(np.ascontiguousarray(image), np.ascontiguousarray(mask), self.mean, self.std)

    def _crop_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        target_class: int | None,
        force_background: bool,
        force_random_crop: bool,
        min_target_pixels: int,
        target_crop_attempts: int,
        foreground_extension_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = _pad_to_size(image, mask, self.height, self.width)
        h, w = mask.shape[:2]

        if force_random_crop:
            y0 = random.randint(0, max(h - self.height, 0))
            x0 = random.randint(0, max(w - self.width, 0))
            return _crop(image, mask, y0, x0, self.height, self.width)

        if force_background:
            for _ in range(24):
                y0 = random.randint(0, max(h - self.height, 0))
                x0 = random.randint(0, max(w - self.width, 0))
                crop_mask = mask[y0 : y0 + self.height, x0 : x0 + self.width]
                valid = crop_mask != IGNORE_INDEX
                foreground = (crop_mask > 0) & valid
                fg_fraction = float(foreground.sum()) / max(float(valid.sum()), 1.0)
                if fg_fraction <= self.background_max_foreground_fraction:
                    return _crop(image, mask, y0, x0, self.height, self.width)

        if target_class is not None:
            ys, xs = np.where(mask == target_class)
        else:
            ys, xs = np.where((mask > 0) & (mask != IGNORE_INDEX))

        if len(ys) > 0:
            best_crop: Tuple[np.ndarray, np.ndarray] | None = None
            best_pixels = -1
            for _ in range(target_crop_attempts):
                point = random.randrange(len(ys))
                jitter_y = random.randint(-self.height // 5, self.height // 5)
                jitter_x = random.randint(-self.width // 5, self.width // 5)
                center_y = int(ys[point]) + jitter_y
                center_x = int(xs[point]) + jitter_x
                crop_h = self.height
                crop_w = self.width
                if foreground_extension_ratio > 0:
                    crop_h = min(max(int(round(self.height * (1.0 + foreground_extension_ratio))), self.height), h)
                    crop_w = min(max(int(round(self.width * (1.0 + foreground_extension_ratio))), self.width), w)
                y0 = min(max(center_y - crop_h // 2, 0), max(h - crop_h, 0))
                x0 = min(max(center_x - crop_w // 2, 0), max(w - crop_w, 0))
                crop = _crop(image, mask, y0, x0, crop_h, crop_w)
                if crop_h != self.height or crop_w != self.width:
                    crop = _resize(crop[0], crop[1], self.height, self.width)
                if target_class is None:
                    target_pixels = int(np.count_nonzero((crop[1] > 0) & (crop[1] != IGNORE_INDEX)))
                else:
                    target_pixels = int(np.count_nonzero(crop[1] == target_class))
                if target_pixels > best_pixels:
                    best_crop = crop
                    best_pixels = target_pixels
                if target_pixels >= min_target_pixels:
                    return crop
            if best_crop is not None:
                return best_crop

        y0 = random.randint(0, max(h - self.height, 0))
        x0 = random.randint(0, max(w - self.width, 0))
        return _crop(image, mask, y0, x0, self.height, self.width)

    def _geometric(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.15:
            image = np.ascontiguousarray(image[::-1])
            mask = np.ascontiguousarray(mask[::-1])

        if random.random() < 0.75:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            scale = random.uniform(self.scale_min, self.scale_max)
            image, mask = _pil_affine(image, mask, angle, scale, self.height, self.width)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

        if random.random() < 0.2 and self.perspective > 0:
            max_dx = self.width * self.perspective
            max_dy = self.height * self.perspective
            src = np.float32([[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]])
            dst = src + np.float32([
                [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)]
                for _ in range(4)
            ])
            try:
                image, mask = _pil_perspective(image, mask, src, dst, self.height, self.width)
            except Exception as exc:
                LOGGER.warning("perspective_augmentation_skipped error=%s", exc)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image)
        image_f = image.astype(np.float32)
        max_value = 65535.0 if image_f.max() > 255 else 255.0
        image_f = image_f / max_value * 255.0

        if random.random() < 0.7:
            image_f += random.uniform(-0.2, 0.2) * 255.0
        if random.random() < 0.7:
            contrast = random.uniform(0.8, 1.2)
            image_f = (image_f - 127.5) * contrast + 127.5
        image_f = np.clip(image_f, 0, 255).astype(np.uint8)

        if random.random() < 0.35:
            gamma = random.uniform(0.8, 1.2)
            table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
            image_f = cv2.LUT(np.ascontiguousarray(image_f), table)
        if random.random() < 0.35:
            image_f = _apply_clahe(np.ascontiguousarray(image_f))
        if random.random() < 0.25:
            noise = np.random.normal(0.0, random.uniform(5.0, 15.0), image_f.shape).astype(np.float32)
            image_f = np.clip(image_f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.12:
            image_f = cv2.GaussianBlur(np.ascontiguousarray(image_f), (3, 3), 0)
        if random.random() < 0.08:
            image_f = _motion_blur(np.ascontiguousarray(image_f))
        return np.ascontiguousarray(image_f)


class EvalTransform:
    """Deterministic resize and normalization."""

    def __init__(self, height: int, width: int, mean: float = 0.5, std: float = 0.25) -> None:
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = _resize(image, mask, self.height, self.width)
        return _to_tensor(image, mask, self.mean, self.std)


def _resize(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if image.ndim == 2:
        image = image[:, :, None]
    return image, mask


def _pad_to_size(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    pad_h = max(height - mask.shape[0], 0)
    pad_w = max(width - mask.shape[1], 0)
    if pad_h == 0 and pad_w == 0:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _crop(image: np.ndarray, mask: np.ndarray, y0: int, x0: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(image[y0 : y0 + height, x0 : x0 + width]),
        np.ascontiguousarray(mask[y0 : y0 + height, x0 : x0 + width]),
    )


def _pil_affine(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float,
    scale: float,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    center_x = width / 2.0
    center_y = height / 2.0
    theta = np.deg2rad(angle)
    cos_t = float(np.cos(theta)) / max(scale, 1e-6)
    sin_t = float(np.sin(theta)) / max(scale, 1e-6)
    coeffs = (
        cos_t,
        sin_t,
        center_x - cos_t * center_x - sin_t * center_y,
        -sin_t,
        cos_t,
        center_y + sin_t * center_x - cos_t * center_y,
    )
    image_pil = Image.fromarray(np.ascontiguousarray(image))
    mask_pil = Image.fromarray(np.ascontiguousarray(mask))
    image_out = image_pil.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    mask_out = mask_pil.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.NEAREST,
        fillcolor=IGNORE_INDEX,
    )
    return (
        np.array(image_out.convert("RGB"), dtype=np.uint8, copy=True),
        np.array(mask_out.convert("L"), dtype=np.uint8, copy=True),
    )


def _pil_perspective(
    image: np.ndarray,
    mask: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = _perspective_coeffs(dst.tolist(), src.tolist())
    image_pil = Image.fromarray(np.ascontiguousarray(image))
    mask_pil = Image.fromarray(np.ascontiguousarray(mask))
    image_out = image_pil.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )
    mask_out = mask_pil.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.NEAREST,
        fillcolor=IGNORE_INDEX,
    )
    return (
        np.array(image_out.convert("RGB"), dtype=np.uint8, copy=True),
        np.array(mask_out.convert("L"), dtype=np.uint8, copy=True),
    )


def _perspective_coeffs(src_points: list[list[float]], dst_points: list[list[float]]) -> tuple[float, ...]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(src_points, dst_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
    return tuple(float(value) for value in coeffs)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2 or image.shape[2] == 1:
        return clahe.apply(image.squeeze(-1))[:, :, None]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _motion_blur(image: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(image)
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if random.random() < 0.5:
        kernel[kernel_size // 2, :] = 1.0
    else:
        kernel[:, kernel_size // 2] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


def _to_tensor(image: np.ndarray, mask: np.ndarray, mean: float, std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.astype(np.float32)
    max_value = 65535.0 if image.max() > 255 else 255.0
    image = image / max_value
    image = (image - mean) / std
    image_t = torch.from_numpy(image).permute(2, 0, 1).float()
    mask_t = torch.from_numpy(mask.astype(np.int64))
    return image_t, mask_t


def _cat_list(tensors, fill_value: int):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    batch_shape = (len(tensors),) + max_size
    batched = tensors[0].new_full(batch_shape, fill_value)
    for src, dst in zip(tensors, batched):
        dst[..., :src.shape[-2], :src.shape[-1]].copy_(src)
    return batched
