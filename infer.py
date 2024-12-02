"""
@author: kaijia.yan@kunlun-inc.com
@date: 2024-12-02
@reference: https://github.com/diptychprompting/diptychprompting.github.io
"""
import warnings
# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
# 忽略UserWarning
warnings.filterwarnings('ignore', category=UserWarning)
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import numpy as np

check_min_version("0.30.2")  # 必须使用这个版本, 否则会出现张量匹配错误

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

size = (2*736, 1152)

Iref_image_path = "your white background image"
Iref_name = os.path.basename(Iref_image_path).split(".")[0].split("_")[-1]
Idiptych_image_path = f"your white diptych image"
Mdiptych_path = f"your white Mdiptych Image"


Iref_image = load_image(Iref_image_path).convert("RGB")
Idiptych_image = load_image(Idiptych_image_path).convert("RGB")
Mdiptych = load_image(Mdiptych_path).convert("RGB")

Object = "girl"
des_left = "dress, brown hair, jewelry, original, long_hair, backless dress, 1girl, earrings, back, from side, ponytail, looking back, sfw, dress_lift, dress pull, ass, female solo, backless, white dress."
des_right = "Full body, with the left hand lifting the hair, smiling from a frontal angle, wearing a wide brimmed hat on the head, with a background of a commercial street, neon lights, and the moon at night"
Tdiptych = f"A diptych with two side-by-side images of the same {Object}. \
On the left, {des_left}. On the right, replicate this {Object} but as {des_right}."
    

seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed)   
# Inpaint
result = pipe(
    prompt=Tdiptych,
    height=size[1],
    width=size[0],
    control_image=Idiptych_image,
    control_mask=Mdiptych,
    num_inference_steps=30,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=3.5,
    negative_prompt="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, disfigured",
    true_guidance_scale=3.5 # default: 3.5 for alpha and 1.0 for beta
).images[0]
save_path = f'output'
os.makedirs(save_path, exist_ok=True)
result.save(f'{save_path}/seed{seed}_{os.path.basename(Iref_image_path).split(".")[0]}.jpg')
print("Successfully inpaint image")
