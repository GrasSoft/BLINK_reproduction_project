import cv2
import numpy as np
import os
import pandas as pd

def find_red_circles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    output_image = image.copy()
    
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of red in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 | mask2
    mask = cv2.medianBlur(mask, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_centers = {
        "A": ((0, 0), 0),
        "B": ((0, 0), 0),
        "A_coords": ((0,0), (0,0)),
        "B_coords": ((0,0), (0,0)),
    }

    # Load templates for A and B
    template_A = cv2.imread('letter_templates/A.png', 0)
    template_B = cv2.imread('letter_templates/B.png', 0)
    
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if radius > 6 and radius < 8:
            center = (int(x), int(y))
            radius_int = int(radius)
            
            # Draw the enclosing circle
            cv2.circle(output_image, center, radius_int, (0, 255, 0), 2)

            roi_size = 50
            x1, y1 = max(int(x) - roi_size, 0), max(int(y) - roi_size, 0)
            x2, y2 = min(int(x) + roi_size, image.shape[1]), min(int(y) + roi_size, image.shape[0])
            roi = image[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            result_A = cv2.matchTemplate(roi_gray, template_A, cv2.TM_CCOEFF_NORMED)
            result_B = cv2.matchTemplate(roi_gray, template_B, cv2.TM_CCOEFF_NORMED)

            threshold = 0.8
            locations_A = np.where(result_A >= threshold)
            locations_B = np.where(result_B >= threshold)

            A_coords = list(zip(*locations_A[::-1]))
            B_coords = list(zip(*locations_B[::-1]))

            margin = 5

            if A_coords:
                detected_centers["A"] = (center, radius)
                for (xa, ya) in A_coords:
                    top_left = (x1 + xa, y1 + ya)
                    bottom_right = (top_left[0] + template_A.shape[1], top_left[1] + template_A.shape[0])
                               
                    top_left_expanded = (max(0, top_left[0] - margin), max(0, top_left[1] - margin))
                    bottom_right_expanded = (
                        min(image.shape[1] - 1, top_left[0] + template_B.shape[1] + margin),
                        min(image.shape[0] - 1, top_left[1] + template_B.shape[0] + margin)
                    )
                                                            
                    detected_centers["A_coords"] = (top_left_expanded, bottom_right_expanded)

                                        
            elif B_coords:
                detected_centers["B"] = (center, radius)
                for (xb, yb) in B_coords:
                    top_left = (x1 + xb, y1 + yb)
                    bottom_right = (top_left[0] + template_B.shape[1], top_left[1] + template_B.shape[0])
                    
                    top_left_expanded = (max(0, top_left[0] - margin), max(0, top_left[1] - margin))
                    bottom_right_expanded = (
                        min(image.shape[1] - 1, top_left[0] + template_B.shape[1] + margin),
                        min(image.shape[0] - 1, top_left[1] + template_B.shape[0] + margin)
                    )
                                                            
                    detected_centers["B_coords"] = (top_left_expanded, bottom_right_expanded)


    return detected_centers

# Example usage:

path = './depth_val_images/'
path_depth = './output_depth_val_images/'
centers = []
for index, item in enumerate(os.listdir(path)):
    full_path = os.path.join(path, item)
    cs = find_red_circles(full_path)

    
    # convert name 
    name, extension = os.path.splitext(item)

    new_name = f"{name}_depth{extension}"
    
    import re
    # extract number from name, e.g. "image_0" → 0
    match = re.search(r'(\d+)', name)
    if not match:
        continue  # skip files with no number
    idx = int(match.group(1))

    
    centers.append({
            "A" : cs["A"],
            "B" : cs["B"],
            "A_coords": cs["A_coords"],
            "B_coords": cs["B_coords"],
            "path_image": full_path,
            "path_depth": os.path.join(path_depth, new_name),
            "idx": idx + 1 ,
        })

    
df_red_circles = pd.DataFrame(centers)

df_questions = pd.read_parquet('../BLINK/Relative_Depth/val-00000-of-00001.parquet', engine='pyarrow')
df_questions['idx'] = df_questions['idx'].str.extract(r'(\d+)').astype(int)

#print(df_questions.head())

merged_df = pd.merge(df_red_circles, df_questions, on='idx', how='inner')

merged_df[["idx", "A", "B", "A_coords", "B_coords", "path_image", "path_depth", "question", "sub_task", "choices", "answer" ]].to_csv("val_depth_with_centers.csv")


