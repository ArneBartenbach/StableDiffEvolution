import os

import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import util
from os import makedirs
import numpy as np

evolution_steps = 100



#vars
prompt="A photorealistic 8k image of a beautiful Roboter"
negative_prompt="ugly, art, blurry, artefacts, abstract"
seed = 0
proj_name = "saveAndRead"
num_inference_steps=50
height = 512
width = 512
model_path = "CompVis/stable-diffusion-v1-4"
torch_dtype=torch.float16
strength=0.2
guidance_scale=7

pop_size = 4
select_every = 5#unused
device = "cuda"



prompt = [prompt] * pop_size
negative_prompt = [negative_prompt] * pop_size

proj_path = "./evolution/"+proj_name+"_"+str(seed)
makedirs(proj_path, exist_ok=True)
makedirs(proj_path+'/selected', exist_ok=True)

generator = torch.Generator(device=device).manual_seed(seed)

for i in range(evolution_steps):
    input_image_path = "D:/Uni/Master/Masterarbeit/Implementierung/StableDiffEvolution/evolution/saveAndRead_0/selected/"+str(len(os.listdir("./evolution/saveAndRead_0/selected/"))-1)+".png"
    torch.manual_seed(seed)  # same seed also set on the created pipe with the generator

    ###############

    init_img = Image.open(input_image_path).convert("RGB")
    init_img = init_img.resize((width, height))

    cur = init_img

    im2im = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
    util.disableNSFWFilter(im2im)

    with autocast("cuda"):
        images = im2im(image=cur, prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, width=width, height=height, generator=generator).images[0]

    cur = images
    cur.save('{}/selected/{}.png'.format(proj_path, str(i+1)))


