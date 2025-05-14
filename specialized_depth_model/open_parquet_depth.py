import os
import pandas as pd
from io import BytesIO
from PIL import Image

# Load your dataset (replace with your actual file path)
df = pd.read_parquet('./BLINK/Relative_Depth/val-00000-of-00001.parquet', engine='pyarrow')

# Specify the output directory where images will be saved
output_directory = 'depth_val_images/'

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Function to save the binary data from 'bytes' field to a .png image
def save_image_from_bytes(image_bytes, image_name):
    image = Image.open(BytesIO(image_bytes))
    # Save as PNG
    image.save(f"{output_directory}{image_name}.png", 'PNG')

# Loop through the dataset, extracting the image bytes from the 'image_1' field
for idx, row in df.iterrows():
    # Extract the image bytes from the 'image_1' field (which is a dictionary with a 'bytes' key)
    image_bytes = row['image_1']['bytes']  # Adjust based on your actual column name
    image_name = f"image_{idx}"  # You can customize the naming logic

    # Save the image
    save_image_from_bytes(image_bytes, image_name)
    print(f"Image {image_name} saved successfully.")

print("All images have been saved.")