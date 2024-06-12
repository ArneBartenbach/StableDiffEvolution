import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
import numpy as np
import torch
from torch import autocast
import util
from PIL import Image


def diff_evoltion(prompt="A beautiful Roboter", negative_prompt="ugly", input_image=None, seed=0, proj_name="defaultprojectName", num_ims=4,
                  num_inference_steps=50, width=512, height=512, weights_path="CompVis/stable-diffusion-v1-4"):
    device = "cuda"

    num_steps = 1   #fixed number of steps - not needed when the function is executed manually n times
    step_size = 0.01
    fill_in_steps = 10 #unused

    torch.manual_seed(seed)
    proj_path = "./evolution/" + proj_name + "_" + str(seed)
    os.makedirs(proj_path, exist_ok=True)
    os.makedirs(proj_path + '/selected', exist_ok=True)

    print('Creating init image')

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(weights_path,torch_dtype=torch.float16,use_auth_token=True).to(device)
    util.disableNSFWFilter(pipe)

    start = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)

    with autocast("cuda"):
        # init_img = pipe(prompt, num_inference_steps=50, latents=start, width=width, height=height)["sample"][0]
        init_img = pipe(prompt, num_inference_steps=50, latents=start, width=width, height=height).images[0]

    init_img.save(proj_path + '/_origin.png')

    cur_latents = torch.cat([start] * num_ims)
    prompt = [prompt] * num_ims

    frame_index = 0
    for i in range(num_steps):
        distant = torch.randn((num_ims, pipe.unet.in_channels, height // 8, width // 8), device=device)
        cur_latents = util.slerp(float(step_size), cur_latents, distant)    #stepsize 0 would results in cur_latents and stepsize 1 would result in distant

        with autocast("cuda"):
            images = pipe(
                prompt=prompt, negative_prompt=negative_prompt, image=input_image,
                num_inference_steps=num_inference_steps,
                latents=cur_latents,
            ).images
        grid_img = util.image_grid(images, 1, num_ims)
        grid_img.save(proj_path + '/cur_pop.png')
        selection = int(input("Select 1-4: "))
        assert selection >= 1 and selection <= 4
        selected_img = images[selection - 1]
        grid_img.save('{}/{}_{}.png'.format(proj_path, str(i), selection))
        selected_img.save('{}/selected/{}.png'.format(proj_path, str(i)))
        cur_latents = torch.stack([cur_latents[selection - 1]] * num_ims, 0)
diff_evoltion()