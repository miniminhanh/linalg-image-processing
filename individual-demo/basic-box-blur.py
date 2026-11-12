import cv2
import numpy as np


# Load the image
image = cv2.imread('image.png')


# Box blur kernel: every weight is equal, so each pixel becomes the plain average
# of the 3x3 square around it. Dividing by 9 (3 * 3) makes the weights sum to 1.
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

# Apply the box blur (-1 keeps the same bit depth as the input image)
blurred_image = cv2.filter2D(image, -1, kernel)

# Display the original and blurred images
cv2.imshow('Original Image', image)
cv2.imshow('Box Blurred Image', blurred_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
