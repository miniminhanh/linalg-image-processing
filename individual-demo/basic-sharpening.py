import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('image.png')

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

sharpened_image = cv2.filter2D(image, -1, kernel)

#showing the image
cv2.imshow('Original', image)
cv2.imshow('Unsharp Masking', sharpened_image)

cv2.waitKey()
cv2.destroyAllWindows()