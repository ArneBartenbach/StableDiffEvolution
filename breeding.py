import os
from torch import autocast

import PIL
import numpy as np
from PIL import Image, ImageFilter

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch
import util
import torchaudio as ta


def initialize(seed=0):
    device = "cuda"
    return torch.Generator(device=device).manual_seed(seed) # just used to set the seed for the pipe

def breed_image(prompt="A photorealistic 8k image of a beautiful Roboter", negative_prompt="ugly, art, blurry, artefacts, abstract", input_image_path_list=None, seed=0,
                proj_name="defaultprojectName", number_of_options=4,
                num_inference_steps=50, width=512, height=512, model_path="CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, strength=0.75, guidance_scale=7.5, step_size=0.01, generator=None): #default values for strength and guidance scale selected from https://getimg.ai/guides/guide-to-stable-diffusion-strength-parameter
    device = "cuda"
    print("Currently Used Config:\n Prompt: ", prompt,"\nNegative_Prompt: ", negative_prompt, "\ninput_image_path: ", input_image_path_list, "\nproj_name: ", proj_name, "\nseed: ", seed,"\nnum_inference_steps: ",num_inference_steps,"\nwidth: ", width,"\nheight: ", height,"\nmodel_path: ", model_path,"\ntorch_dtype: ", torch_dtype,"\nstrength: ", strength,"\nguidance_scale: ", guidance_scale, "\nstep_size: ",step_size)
    out_paths = []


    torch.manual_seed(seed) #same seed also set on the created pipe with the generator

    promptx = [prompt]*number_of_options
    negative_promptx = [negative_prompt]*number_of_options

    proj_path = "./breed/" + proj_name + "_" + str(seed)
    os.makedirs(proj_path, exist_ok=True)

    if input_image_path_list is None: #first image needs to be created just from the prompt and negative prompt - StableDiffusionPipeline is used for text to image
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
        util.disableNSFWFilter(pipe)
        with autocast("cuda"):
            init_img = pipe(prompt=promptx, negative_prompt=negative_promptx, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height, generator=generator).images
        for img in init_img:
            img.save(proj_path+"/"+ str(len(os.listdir(proj_path))) +".png")
            out_paths.append(proj_path+"/"+ str(len(os.listdir(proj_path))-1) +".png")



    #        init_img.save(proj_path+"/"+ str(len(os.listdir(proj_path))) +".png")


    else: #images is created based on the combination of image and text - StableDiffusionImg2ImgPipeline is used for text and image to image
        input_image_list = []
        for input_image_path in input_image_path_list:
            input_image = Image.open(input_image_path).convert('RGB')


            noisy_image = input_image#addNoise(input_image) #TODO changed

            #gaussian filter tested to reduce convergence into abstract image
            #input_image = input_image.filter(ImageFilter.GaussianBlur(radius=20)).convert("RGB")

            noisy_image = noisy_image.resize((width, height))
            input_image_list.append(noisy_image)

        #create im2im-Pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
        util.disableNSFWFilter(pipe)

        ####################################################################################
        # adding random onto the latents
        #num_ims = 1
        #distant = torch.randn((num_ims, pipe.unet.in_channels, height // 8, width // 8), device=device) #unet.in_channels are most often  4 (3RGB and one additional for e.g. latent noise
        #util.slerp(float(step_size), cur_latents, distant)
        ####################################################################################
        with autocast("cuda"):
            img_list = pipe(image=input_image_list, prompt=promptx, negative_prompt=negative_promptx, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, width=width, height=height, generator=generator).images
        for img_iter in img_list:
            img_iter.save(proj_path+"/"+ str(len(os.listdir(proj_path))) +".png")
            out_paths.append(proj_path+"/"+ str(len(os.listdir(proj_path))-1) +".png")

    return out_paths



#breed_image()
#evolution_steps=100
#for i in range(evolution_steps):
#    path = "D:/Uni/Master/Masterarbeit/Implementierung/StableDiffEvolution/breed/defaultprojectName_0/"+str(len(os.listdir("./breed/defaultprojectName_0/"))-1)+".png"
#    breed_image(input_image_path=path)
path1 = "C:/Users/Arne/Desktop/38.png"
path2 = "C:/Users/Arne/Desktop/46.png"
breed_image(prompt="clear image, straight lines",input_image_path_list=[path1, path2], strength=0.9)

def addNoise(input_image):
    img_tensor = torch.from_numpy(np.array(input_image) / 255.0).float()
    noise_factor = 0.1
    noise = torch.randn(img_tensor.size()) * noise_factor
    noisy_img_tensor = img_tensor + noise
    noisy_img_tensor = noisy_img_tensor.clamp(0.0, 1.1)
    noisy_img_array = noisy_img_tensor.cpu().detach().numpy()
    noisy_img_array = noisy_img_array * 255
    noisy_image = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_image