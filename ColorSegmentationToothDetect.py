import cv2
import numpy as np

# Load image
img = cv2.imread('input_image.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV range for white toothpaste tube body (adjust for your lighting)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Mask for white regions
mask = cv2.inRange(hsv, lower_white, upper_white)

# Morphological operations to clean mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

toothpaste_found = False
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 5000:  # Min area threshold for tube
        # Approximate contour to polygon (cylinder ~ rectangle in 2D)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4 and len(approx) <= 8:  # Roughly rectangular/cylindrical
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 0.8:  # Typical tube proportions
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, 'Toothpaste Tube', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                toothpaste_found = True
                print("Toothpaste detected at:", (x, y, w, h))

if not toothpaste_found:
    print("No toothpaste tube found.")

# Display results
cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_contour.jpg', img)