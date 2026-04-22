"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import numpy as np
from typing import Optional, Tuple

def sample_factor(bounds):
    return np.random.uniform(bounds[0], bounds[1])

def apply_brightness(a: np.ndarray, factor: float) -> np.ndarray:
    return a * factor

def apply_contrast(a: np.ndarray, factor: float) -> np.ndarray:
    mean = a.mean(axis=(0, 1), keepdims=True)
    return (a - mean) * factor + mean

def rgb_to_gray(a: np.ndarray) -> np.ndarray:
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

def apply_saturation(a: np.ndarray, factor: float) -> np.ndarray:
    gray = rgb_to_gray(a)[..., None]
    return gray + (a - gray) * factor

def rgb_to_hsv(a: np.ndarray) -> np.ndarray:
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 0
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask

    h[r_eq] = ((g - b)[r_eq] / diff[r_eq]) % 6.0
    h[g_eq] = ((b - r)[g_eq] / diff[g_eq]) + 2.0
    h[b_eq] = ((r - g)[b_eq] / diff[b_eq]) + 4.0
    h = h / 6.0

    s = np.zeros_like(mx)
    s[mx > 0] = diff[mx > 0] / mx[mx > 0]

    v = mx
    return np.stack([h, s, v], axis=-1)

def hsv_to_rgb(a: np.ndarray) -> np.ndarray:
    h, s, v = a[..., 0], a[..., 1], a[..., 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.choose(i % 6, [v, q, p, p, t, v])
    g = np.choose(i % 6, [t, v, v, q, p, p])
    b = np.choose(i % 6, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1)

def random_photometric_distortion(
    img_np: np.ndarray,
    brightness: Optional[Tuple[float, float]] = (0.8, 1.2),
    contrast: Optional[Tuple[float, float]] = (0.8, 1.2),
    saturation: Optional[Tuple[float, float]] = (0.8, 1.2),
    hue: Optional[Tuple[float, float]] = (-0.1, 0.1),
    p: float = 0.5,
) -> np.ndarray:
    if p == 0:
        return img_np


    rng = np.random.default_rng()

    info = np.iinfo(img_np.dtype)
    vmin, vmax = max(0, info.min), info.max

    x = img_np.astype(np.float32)

    ops = []
    if brightness is not None:
        ops.append(("brightness", brightness, apply_brightness))
    if contrast is not None:
        ops.append(("contrast", contrast, apply_contrast))
    if saturation is not None:
        ops.append(("saturation", saturation, apply_saturation))
    if hue is not None:
        ops.append(("hue", hue, None))

    rng.shuffle(ops)

    for name, bounds, fn in ops:
        if rng.random() >= p:
            continue

        if name == "hue":
            x = x.clip(0, 100000)
            factor = sample_factor(bounds)
            x01 = np.clip((x - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            hsv = rgb_to_hsv(x01)
            hsv[..., 0] = (hsv[..., 0] + factor) % 1.0
            x01 = hsv_to_rgb(hsv)
            x = x01 * (vmax - vmin) + vmin
        else:
            factor = sample_factor(bounds)
            x = fn(x, factor)

    return np.clip(x, vmin, vmax).round().astype(img_np.dtype)
