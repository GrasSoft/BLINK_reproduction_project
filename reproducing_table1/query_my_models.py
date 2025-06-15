import requests
import base64

"""
Includes functions to query LLaVA and BakLLaVA vision-language models via the Ollama API.
Assumes Ollama is running at http://localhost:11434.
"""


def encode_image(image_path):
    """
       Encode an image file as a base64 string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def query_llava(image_paths, prompt):
    """
    Query the LLaVA model with the prompt and a list of image paths.
    Assumes image_paths contains only one image (possibly concatenated).
    """
    if len(image_paths) == 0:
        raise ValueError("No image path provided to LLaVA.")

    # LLaVA only accepts one image; raise an error if multiple are provided.
    if len(image_paths) > 1:
        raise ValueError(f"LLaVA received multiple images: {image_paths}. Concatenate them before calling.")

    image_path = image_paths[0]
    encoded_image = encode_image(image_path)

    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }

    # Send request to Ollama API (LLaVA/BakLLaVA must be running locally in Ollama).
    response = requests.post("http://localhost:11434/api/generate", json=payload)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        print("LLaVA error:", response.status_code, response.text)
        return ""


def query_bakllava(image_paths, prompt):
    """
    Query the BakLLaVA model with the prompt and a list of image paths.
    """
    if len(image_paths) == 0:
        raise ValueError("No image path provided to BakLLaVA.")

    if len(image_paths) > 1:
        raise ValueError(f"BakLLaVA received multiple images: {image_paths}. Concatenate them before calling.")

    image_path = image_paths[0]
    encoded_image = encode_image(image_path)

    payload = {
        "model": "bakllava",
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }

    # Send request to Ollama API (LLaVA/BakLLaVA must be running locally in Ollama).
    response = requests.post("http://localhost:11434/api/generate", json=payload)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        print("BakLLaVA error:", response.status_code, response.text)
        return ""