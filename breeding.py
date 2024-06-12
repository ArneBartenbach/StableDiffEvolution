import os

import PIL
import numpy as np
from PIL import Image, ImageFilter

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch
import util
import torchaudio as ta

def breed_image(toggle=1, prompt="A photorealistic 8k image of a beautiful Roboter", negative_prompt="ugly, art, blurry, artefacts, abstract", input_image_path=None, seed=0,
                proj_name="defaultprojectName", num_ims=4,
                num_inference_steps=50, width=512, height=512, weights_path="CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, strength=0.2, guidance_scale=7 , step_size=0.01): #default values for strength and guidance scale selected from https://getimg.ai/guides/guide-to-stable-diffusion-strength-parameter
    device = "cuda"
    torch.manual_seed(seed) #same seed also set on the created pipe with the generator
    generator = torch.Generator(device=device).manual_seed(seed) # just used to set the seed for the pipe

    #generator #parameter most likely not requiredm since in the examples the manual generator is only used to set a seed which can be done directly on the pipe

    proj_path = "./breed/" + proj_name + "_" + str(seed)
    os.makedirs(proj_path, exist_ok=True)

    if input_image_path is None: #first image needs to be created just from the prompt and negative prompt - StableDiffusionPipeline is used for text to image
        pipe = StableDiffusionPipeline.from_pretrained(weights_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
        util.disableNSFWFilter(pipe)
        init_img = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale = guidance_scale, width=width, height=height, generator=generator).images[0]
        print(type(init_img))
        print(init_img)
        init_img.save(proj_path+"/"+ str(len(os.listdir(proj_path))) +".png")


    else: #images is created based on the combination of image and text - StableDiffusionImg2ImgPipeline is used for text and image to image
        input_image = Image.open(input_image_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(input_image) / 255.0).float()
        noise_factor = 0.1
        noise = torch.randn(img_tensor.size()) * noise_factor

        if toggle == 1:
            noisy_img_tensor = img_tensor + noise
        else:# do nothing
            noisy_img_tensor = img_tensor

        noisy_img_tensor = noisy_img_tensor.clamp(0.0, 1.1)
        noisy_img_array = noisy_img_tensor.cpu().detach().numpy()
        noisy_img_array = noisy_img_array * 255
        input_image = Image.fromarray(noisy_img_array.astype(np.uint8))
        #gaussian filter tested to reduce convergence into abstract image
        #input_image = input_image.filter(ImageFilter.GaussianBlur(radius=20)).convert("RGB")

        input_image = input_image.resize((width, height))

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(weights_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
        util.disableNSFWFilter(pipe)

        ####################################################################################
        # adding random onto the latents
        #num_ims = 1
        #distant = torch.randn((num_ims, pipe.unet.in_channels, height // 8, width // 8), device=device) #unet.in_channels are most often  4 (3RGB and one additional for e.g. latent noise
        #util.slerp(float(step_size), cur_latents, distant)
        ####################################################################################

        img = pipe(image=input_image, prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, strength=strength, guidance_scale = guidance_scale, width=width, height=height, generator=generator).images[0]
        print(type(img))
        print(img)
        img.save(proj_path+"/"+ str(len(os.listdir(proj_path))) +".png")

breed_image()
x=0
for i in range(100):
    breed_image(toggle=x,input_image_path="D:/Uni/Master/Masterarbeit/Implementierung/StableDiffEvolution/breed/defaultprojectName_0/"+str(len(os.listdir("./breed/defaultprojectName_0/"))-1)+".png")
    if x == 1:
        x = 0
    else:
        x = 1