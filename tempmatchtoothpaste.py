import cv2
import numpy as np

# Load input image and template (save a cropped toothpaste tube as 'template.jpg')
input_img = cv2.imread('input_image.jpg')
template = cv2.imread('template.jpg')

# Get template dimensions
h, w = template.shape[:2]

# Perform template matching (normalized method for scale/illumination invariance)
result = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Threshold for good match (adjust 0.8 as needed)
if max_val >= 0.8:
    # Draw rectangle around detected toothpaste
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(input_img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(input_img, 'Toothpaste Detected!', (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print("Toothpaste detected at:", top_left, bottom_right)
else:
    print("No toothpaste match found.")

# Display result
cv2.imshow('Detection Result', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_template.jpg', input_img)