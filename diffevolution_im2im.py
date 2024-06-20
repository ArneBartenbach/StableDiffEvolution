import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import util
from os import makedirs
import numpy as np

init_img_path = "D:/Uni/Master/Masterarbeit/Implementierung/StableDiffEvolution/breed/defaultprojectName_0/0.png"
prompt = "A photorealistic 8k image of a beautiful Roboter"
proj_name = "saveAndRead"
seed = 0

pop_size = 4
evolution_steps = 100
select_every = 5#unused
height = 512
width = 512

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"
generator = torch.Generator(device=device).manual_seed(seed)
proj_path = "./evolution/"+proj_name+"_"+str(seed)
makedirs(proj_path, exist_ok=True)
makedirs(proj_path+'/selected', exist_ok=True)


if init_img_path is None:
    print('Creating init image')
    text2im = StableDiffusionPipeline.from_pretrained(
        model_path,
        use_auth_token=True
    ).to(device)
    util.disableNSFWFilter(text2im)
    with autocast("cuda"):
        init_img = text2im(prompt, num_inference_steps=50, width=width, height=height, generator=generator)["sample"][0]
    del text2im
    torch.cuda.empty_cache()
else:
    init_img = Image.open(init_img_path).convert("RGB")
    init_img = init_img.resize((width, height))

init_img.save(proj_path+'/_origin.png')

prompt = [prompt]*pop_size
negative_prompt = ["ugly, art, blurry, artefacts, abstract"]*pop_size
cur = init_img

for i in range(evolution_steps):

    im2im = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        use_auth_token=True,
        torch_dtype=torch.float16
    ).to(device)
    util.disableNSFWFilter(im2im)

    with autocast("cuda"):
        images = im2im(image=cur, prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, strength=0.2, guidance_scale=7, width=width, height=height, generator=generator).images
    image = util.image_grid(images, 1, pop_size)
    image.save(proj_path+'/cur_pop.png')
    selection = 1#int(input("Select 1-4: "))
    assert selection >= 1 and selection <= 4
    cur = images[selection-1]
    image.save('{}/{}_{}.png'.format(proj_path, str(i), selection))
    cur.save('{}/selected/{}.png'.format(proj_path, str(i)))
    cur = Image.open('{}/selected/{}.png'.format(proj_path, str(i))).convert('RGB')
    cur = cur.resize((width, height))

    torch.cuda.empty_cache()



