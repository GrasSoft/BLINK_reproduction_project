import os 
import pandas as pd
import cv2
import numpy as np
import ast

df = pd.read_csv("val_depth_with_centers.csv")

def is_closer(image_path, circle_A, circle_B):    
    image = cv2.imread(image_path, 0)

    circle_A = ast.literal_eval(circle_A)
    circle_B = ast.literal_eval(circle_B)

    # Extract circle parameters
    coordA, rA = circle_A
    coordB, rB = circle_B

    (xA, yA) = coordA
    (xB, yB) = coordB


    # Create masks for the circles
    mask_A = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask_A, (int(xA), int(yA)), int(rA), 255, -1)

    mask_B = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask_B, (int(xB), int(yB)), int(rB), 255, -1)

    # Mask the image to get the region inside each circle
    region_A = cv2.bitwise_and(image, image, mask=mask_A)
    region_B = cv2.bitwise_and(image, image, mask=mask_B)

    # Compute the average depth (intensity) in each region
    avg_A = np.mean(region_A[region_A > 0])  # Ignore black pixels (0)
    avg_B = np.mean(region_B[region_B > 0])  # Ignore black pixels (0)

    # Return True if A is farther than B (i.e., A has a higher average intensity)
    return avg_A > avg_B
    
    # just the circle center points dont work that well
    
    # # Get the pixel values at the center of each circle
    # value_A = image[int(yA), int(xA)]  # y, x order
    # value_B = image[int(yB), int(xB)]  # y, x order

    # # Compare the values: A is farther if A has a higher intensity (since whiter is farther)
    # return value_A > value_B
    

count = 0
count_correct = 0
tc = 0
for index, row in df.iterrows():
    if (is_closer(row["path_depth"], row["A"], row["B"]) and ("A" in row["answer"]) ) or (not is_closer(row["path_depth"], row["A"], row["B"]) and ("B" in row["answer"])) and not ((0, 0) in ast.literal_eval(row["A"]) or (0, 0) in ast.literal_eval(row["B"])):
        count_correct += 1
        
    if not ((0, 0) in ast.literal_eval(row["A"]) or (0, 0) in ast.literal_eval(row["B"])):
        
        count += 1
    tc += 1
    
    
print(f"There were this many correct guesses: {count_correct}")
print(f"There were this many items (to assess): {count}")
print(f"There were this many items (total): {tc}")
print(f"Accuracy: {count_correct / count}")