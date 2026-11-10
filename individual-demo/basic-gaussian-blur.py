import cv2
import numpy as np


# Load the image
image = cv2.imread('image.png')


# Apply Gaussian blur
# (3, 3) is the kernel size, and 0 is the standard deviation (calculated automatically when set as default 0)
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Display the original and blurred images
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blurred Image', blurred_image)

cv2.waitKey(0)
cv2.destroyAllWindows()