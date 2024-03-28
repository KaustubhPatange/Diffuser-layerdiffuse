import os
from urllib.parse import urlparse

import cv2
import numpy as np
import safetensors.torch
import torch
from torch.hub import download_url_to_file

import lib_layerdiffusion.pickle
from lib_layerdiffusion.enums import ResizeMode


def get_torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def forge_clip_encode(clip, text):
    if text is None:
        return None

    tokens = clip.tokenize(text, return_word_ids=True)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond.to(get_torch_device())


def rgba2rgbfp32(x):
    rgb = x[..., :3].astype(np.float32) / 255.0
    a = x[..., 3:4].astype(np.float32) / 255.0
    return 0.5 + (rgb - 0.5) * a


def to255unit8(x):
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def safe_numpy(x):
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def high_quality_resize(x, size):
    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        if (size[0] * size[1]) < (x.shape[0] * x.shape[1]):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4

        y = cv2.resize(x, size, interpolation=interpolation)
    else:
        y = x
    return y


def crop_and_resize_image(detected_map, resize_mode, h, w):
    if resize_mode == ResizeMode.RESIZE:
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == ResizeMode.RESIZE_AND_FILL:
        k = min(k0, k1)
        borders = np.concatenate(
            [
                detected_map[0, :, :],
                detected_map[-1, :, :],
                detected_map[:, 0, :],
                detected_map[:, -1, :],
            ],
            axis=0,
        )
        high_quality_border_color = np.median(borders, axis=0).astype(
            detected_map.dtype
        )
        high_quality_background = np.tile(
            high_quality_border_color[None, None], [h, w, 1]
        )
        detected_map = high_quality_resize(
            detected_map, (safeint(old_w * k), safeint(old_h * k))
        )
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[
            pad_h : pad_h + new_h, pad_w : pad_w + new_w
        ] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(
            detected_map, (safeint(old_w * k), safeint(old_h * k))
        )
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h : pad_h + h, pad_w : pad_w + w]
        detected_map = safe_numpy(detected_map)
        return detected_map


def pytorch_to_numpy(x):
    return [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:
                print(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
                )
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(
                ckpt,
                map_location=device,
                pickle_module=lib_layerdiffusion.pickle,
            )
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def is_device_mps(device):
    if hasattr(device, "type"):
        if device.type == "mps":
            return True
    return False


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    return True


def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))
    del prev


def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)
