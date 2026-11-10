import cv2
import numpy as np


# Load the image and convert to grayscale (edge detection works on intensity)
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth first so noise isn't picked up as edges
blurred = cv2.GaussianBlur(gray, (3, 3), 0)


# Sobel kernels: approximate the image gradient in x and y
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
sobel_y_kernel = sobel_x_kernel.T  # the y kernel is the transpose of the x kernel

# Convolve with each kernel (CV_64F keeps negative gradient values)
grad_x = cv2.filter2D(blurred, cv2.CV_64F, sobel_x_kernel)
grad_y = cv2.filter2D(blurred, cv2.CV_64F, sobel_y_kernel)

# Gradient magnitude = sqrt(gx^2 + gy^2), scaled back to 0-255 for display
magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
sobel_edges = cv2.convertScaleAbs(magnitude / magnitude.max() * 255)


# Canny edge detection: gradients + non-maximum suppression + hysteresis thresholding
# 100 and 200 are the lower and upper thresholds
canny_edges = cv2.Canny(blurred, 100, 200)


# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Sobel Edges', sobel_edges)
cv2.imshow('Canny Edges', canny_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
