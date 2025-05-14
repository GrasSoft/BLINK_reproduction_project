import cv2
import numpy as np
import os
import pandas as pd

def find_red_circles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
        
    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Create a mask for red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Define another range of red color (since red spans the beginning and end of the hue circle)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create a second mask for the second red range
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both masks
    mask = mask1 | mask2
            
    mask = cv2.medianBlur(mask,5)
        
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_centers = {
        "A": ((0,0), 0),
        "B": ((0,0), 0),
    }
        
    # Iterate through each contour
    for contour in contours:
        # Get the bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Only consider the contour if it is approximately circular
        if radius > 4 and radius < 20 :  # Adjust size based on expected circle size            
            # Define a Region of Interest (ROI) around the circle's center
            roi_size = 50  # Adjust based on the size of your circles
            x1, y1 = max(int(x) - roi_size, 0), max(int(y) - roi_size, 0)
            x2, y2 = min(int(x) + roi_size, image.shape[1]), min(int(y) + roi_size, image.shape[0])

            roi = image[y1:y2, x1:x2]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Load template images for "A" and "B" (create these from the image if needed)
            template_A = cv2.imread('letter_templates/A.png', 0)  # You can create this manually or extract from the image
            template_B = cv2.imread('letter_templates/B.png', 0)
            
            # Use template matching to find the letters "A" and "B"
            result_A = cv2.matchTemplate(roi, template_A, cv2.TM_CCOEFF_NORMED)
            result_B = cv2.matchTemplate(roi, template_B, cv2.TM_CCOEFF_NORMED)

            # Set a threshold for matching
            threshold = 0.8
            locations_A = np.where(result_A >= threshold)
            locations_B = np.where(result_B >= threshold)

            # Convert the locations into a list of coordinates
            A_coords = list(zip(*locations_A[::-1]))
            B_coords = list(zip(*locations_B[::-1]))

            if A_coords:
                detected_centers["A"] = ((x, y), radius)
            if B_coords:
                detected_centers["B"] = ((x, y), radius)

    return detected_centers

# Example usage:

path = '/home/gras/Documents/University/Generative Modeling/depth_val_images/'
path_depth = '/home/gras/Documents/University/Generative Modeling/output_depth_val_images/'
centers = []
for index, item in enumerate(os.listdir(path)):
    full_path = os.path.join(path, item)
    cs = find_red_circles(full_path)

    
    # convert name 
    name, extension = os.path.splitext(item)

    new_name = f"{name}_depth{extension}"
    
    centers.append({
            "A" : cs["A"],
            "B" : cs["B"],
            "path_image": full_path,
            "path_depth": os.path.join(path_depth, new_name),
            "idx": index,
        })

    
df_red_circles = pd.DataFrame(centers)

df_questions = pd.read_parquet('./BLINK/Relative_Depth/val-00000-of-00001.parquet', engine='pyarrow')
df_questions['idx'] = df_questions['idx'].str.extract(r'(\d+)').astype(int)

merged_df = pd.merge(df_red_circles, df_questions, on='idx', how='inner')

merged_df[["idx", "A", "B", "path_image", "path_depth", "question", "sub_task", "choices", "answer" ]].to_csv("val_depth_with_centers.csv")


