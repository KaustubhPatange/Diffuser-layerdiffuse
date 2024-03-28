import gc
import os

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from imageprocessor import VaeImageProcessor
from lib_layerdiffusion.models import TransparentVAEDecoder
from lib_layerdiffusion.patcher import UnetPatcher
from lib_layerdiffusion.utils import (
    get_torch_device,
    load_file_from_url,
    load_torch_file,
)


class VaeProcessingOutput:
    extra_result_images = []


device = get_torch_device()


def patch_pipe(pipe, vae_po, weight=1.0):
    layer_model_root = os.path.join(os.path.expanduser("~"), ".cache", "layer_model")
    os.makedirs(layer_model_root, exist_ok=True)

    model_path = load_file_from_url(
        url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
        model_dir=layer_model_root,
        file_name="vae_transparent_decoder.safetensors",
    )
    vae_transparent_decoder = TransparentVAEDecoder(load_torch_file(model_path))

    decoder_wrapper = vae_transparent_decoder.decode_wrapper(vae_po)

    decode_old = pipe.vae.decode

    def decode(z: torch.FloatTensor, return_dict: bool = True, generator=None):
        return decoder_wrapper(decode_old, z)

    pipe.vae.decode = decode

    model_path = load_file_from_url(
        url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors",
        model_dir=layer_model_root,
        file_name="vae_transparent_encoder.safetensors",
    )

    model_path = load_file_from_url(
        url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
        model_dir=layer_model_root,
        file_name="layer_xl_transparent_attn.safetensors",
    )
    layer_lora_model = load_torch_file(model_path, safe_load=True)

    unet_patcher = UnetPatcher(pipe.unet, device)
    unet_patcher.load_frozen_patcher(layer_lora_model, weight)
    unet = unet_patcher.patch_model(device_to=device)
    pipe.unet = unet


pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    torch_dtype=torch.float16,
    variant="fp16",
    use_saftensors=True,
).to(device)
pipe.image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

vae_po = VaeProcessingOutput()
patch_pipe(pipe, vae_po)

gen = torch.Generator().manual_seed(1234)

images = pipe.__call__(
    prompt="a glass bottle, high quality",
    negative_prompt="bad, ugly",
    num_inference_steps=30,
    width=1024,
    height=1024,
    generator=gen,
).images

# intermediate first output
images[0].save("out_intermediate.png")

# final output after transparent vae decode
Image.fromarray(vae_po.extra_result_images[0]).convert("RGBA").save("out_final.png")
