import torch
from flask import Flask, request, jsonify
import os  # Optional for image path handling
from flask_cors import CORS

import breeding


generator = None
# Replace with your Stable Diffusion generation function (modify as needed)


def generate_image(proj_name, image, prompt, negative_prompt, number_of_options, seed, guidance_scale, strength, model_path, num_inference_steps, width, height, torch_dtype):
  return breeding.breed_image(proj_name=proj_name, input_image_path=image, prompt=prompt, negative_prompt=negative_prompt, number_of_options=number_of_options, seed=seed, guidance_scale=guidance_scale, strength=strength, model_path=model_path, num_inference_steps=num_inference_steps, width=width, height=height, torch_dtype=torch_dtype, generator=generator)


app = Flask(__name__)
CORS(app)  # Apply CORS for all routes

@app.route('/evolution', methods=['POST'])
def generate_and_return_image():
  data = request.get_json()
  if not data:
    return jsonify({'error': 'Missing request data'}), 400  # Bad request

  image = data.get('inImagePath')
  prompt = data.get('prompt')
  negative_prompt = data.get('negativePrompt')
  number_of_options = int(data.get('numberOfOptions'))
  seed = int(data.get('seed'))
  guidance_scale = float(data.get('guidance_scale'))
  strength = float(data.get('strength'))
  model_path = data.get('model_path')
  num_inference_steps = int(data.get('inferenceSteps'))
  width = int(data.get('width'))
  height = int(data.get('height'))
  torch_dtype_string = data.get('torch_dtype')
  proj_name = data.get('proj_name')
  if torch_dtype_string == "torch.float16":
    torch_dtype = torch.float16
  elif torch_dtype_string == "torch.float32":
    torch_dtype_string = torch.float32
  else:
    torch_dtype = "auto"


  #use this depending on the desired outcome when not prompt is added
  #if not prompt:
  #  return jsonify({'error': 'Missing prompt in data'}), 400

  try:
    image_paths = generate_image(proj_name, image, prompt, negative_prompt, number_of_options, seed, guidance_scale, strength, model_path, num_inference_steps, width, height, torch_dtype)
    abs_image_paths = []
    for image_path in image_paths:
      abs_image_paths.append(os.path.abspath(image_path))
    print(abs_image_paths)
    # Return image URL (adjust based on your image storage)
    return jsonify({'imageUrls': abs_image_paths})

  except Exception as e:
    print(f"Error during image generation: {e}")
    return jsonify({'error': 'Internal server error'}), 500  # Internal server error

if __name__ == '__main__':
  #TODO - remove hardcoded generator init
  generator = breeding.initialize(seed=0)


  app.run(debug=True, host='0.0.0.0', port=8085)  # Run on localhost:8080

#generate_image(None, "X","Y")