import base64

import requests


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# def query_gpt4v(image_paths, prompt, retry=10):
#     """
#     Query the GPT-4 Vision model with the prompt and a list of image paths. The temperature is set to 0.0 and retry is set to 10 if fails as default setting.

#     Parameters:
#     - image_paths: List of Strings, the path to the images.
#     - prompt: String, the prompt.
#     - retry: Integer, the number of retries.
#     """
#     base64_images = [encode_image(image_path) for image_path in image_paths]

#     for r in range(retry):
#         try:
#             input_dicts = [{"type": "text", "text": prompt}]
#             for i, image in enumerate(base64_images):
#                 input_dicts.append(
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{image}",
#                             "detail": "low",
#                         },
#                     }
#                 )
#             response = client.chat.completions.create(
#                 model="gpt-4-vision-preview",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": input_dicts,
#                     }
#                 ],
#                 max_tokens=1024,
#                 n=1,
#                 temperature=0.0,
#             )
#             print(response)
#             return response.choices[0].message.content
#         except Exception as e:
#             print(e)
#             time.sleep(1)
#     return "Failed: Query GPT4V Error"


def query_ollama(image_paths, prompt: str, model_name: str):
    url = "http://localhost:11434/api/generate"
    base64_images = [encode_image(image_path) for image_path in image_paths]
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model_name,
        "prompt": prompt,
        "images": base64_images,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["response"]
    else:
        print("Error:", response.status_code, response.text)
        return None
