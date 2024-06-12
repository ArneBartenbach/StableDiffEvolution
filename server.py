from flask import Flask, request, jsonify
import os  # Optional for image path handling
from flask_cors import CORS
import diffevolution

# Replace with your Stable Diffusion generation function (modify as needed)
def generate_image(prompt):
  return diffevolution.diff_evoltion()


app = Flask(__name__)
CORS(app)  # Apply CORS for all routes

@app.route('/evolution', methods=['POST'])
def generate_and_return_image():
  data = request.get_json()

  if not data:
    return jsonify({'error': 'Missing request data'}), 400  # Bad request

  prompt = data.get('prompt')

  if not prompt:
    return jsonify({'error': 'Missing prompt in data'}), 400

  try:
    image_path = generate_image(prompt)

    # Return image URL (adjust based on your image storage)
    image_url = os.path.join('C:/Users/Arne/Desktop/', image_path)  # Example URL structure
    return jsonify({'imageUrl': image_url})

  except Exception as e:
    print(f"Error during image generation: {e}")
    return jsonify({'error': 'Internal server error'}), 500  # Internal server error

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=8085)  # Run on localhost:8080
