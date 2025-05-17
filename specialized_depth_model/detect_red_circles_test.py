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

            if A_coords:
                detected_centers["A"] = (center, radius)
                for (xa, ya) in A_coords:
                    top_left = (x1 + xa, y1 + ya)
                    bottom_right = (top_left[0] + template_A.shape[1], top_left[1] + template_A.shape[0])
                    cv2.rectangle(output_image, top_left, bottom_right, (255, 0, 0), 2)
                    cv2.putText(output_image, 'A', (top_left[0], top_left[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                        
            elif B_coords:
                detected_centers["B"] = (center, radius)
                for (xb, yb) in B_coords:
                    top_left = (x1 + xb, y1 + yb)
                    bottom_right = (top_left[0] + template_B.shape[1], top_left[1] + template_B.shape[0])
                    cv2.rectangle(output_image, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.putText(output_image, 'B', (top_left[0], top_left[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show the image with detected circles and letters
    cv2.imshow("Detected Circles and Letters", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_centers

# Example usage:

path = './depth_val_images/image_70.png'

cs = find_red_circles(path)




