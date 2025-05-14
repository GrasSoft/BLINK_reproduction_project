import os
import pandas as pd
from io import BytesIO
from PIL import Image

# Load your dataset (replace with your actual file path)
df = pd.read_parquet('./BLINK/Art_Style/val-00000-of-00001.parquet', engine='pyarrow')

# Specify the output directory where images will be saved
output_original = 'style_original_images/'
output_option_A = 'style_A_images/'
output_option_B = 'style_B_images/'

# Create the output directory if it does not exist
os.makedirs(output_original, exist_ok=True)
os.makedirs(output_option_A, exist_ok=True)
os.makedirs(output_option_B, exist_ok=True)


# Function to save the binary data from 'bytes' field to a .png image
def save_image_from_bytes(output_directory, image_bytes, image_name):
    image = Image.open(BytesIO(image_bytes))
    # Save as PNG
    image.save(f"{output_directory}{image_name}.png", 'PNG')

# Loop through the dataset, extracting the image bytes from the 'image_1' field
for idx, row in df.iterrows():
    # Extract the image bytes from the 'image_1' field (which is a dictionary with a 'bytes' key)
    image_original = row['image_1']['bytes']  # Adjust based on your actual column name
    image_A = row['image_2']['bytes']
    image_B = row['image_3']['bytes']
    image_name = f"image_{idx}"  # You can customize the naming logic

    # Save the images
    save_image_from_bytes(output_original, image_original, image_name)
    save_image_from_bytes(output_option_A, image_A, image_name)
    save_image_from_bytes(output_option_B, image_B, image_name)

    print(f"Images {image_name} saved successfully.")

print("All images have been saved.")