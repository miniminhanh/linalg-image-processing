import cv2
import numpy as np


# Load the image as grayscale floats in [0, 1]
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0


# Build the Gaussian blur kernel (the "K" in b = Kx)
# A bigger kernel than basic-gaussian-blur.py so the blur is clearly visible
kernel_size = 9
sigma = 2.0
g = cv2.getGaussianKernel(kernel_size, sigma)
kernel = g @ g.T  # 2D Gaussian = outer product of two 1D Gaussians


# Move the kernel into the frequency domain
# Pad it to the image size and shift its centre to (0, 0) so the FFT lines up
padded_kernel = np.zeros_like(image)
padded_kernel[:kernel_size, :kernel_size] = kernel
padded_kernel = np.roll(padded_kernel, (-(kernel_size // 2), -(kernel_size // 2)), axis=(0, 1))
H = np.fft.fft2(padded_kernel)


# Blur the image: convolution becomes element-wise multiplication in the frequency domain
blurred = np.real(np.fft.ifft2(np.fft.fft2(image) * H))

# Save to 8-bit like a real image file would; this rounding is the only "noise" we add
blurred = np.round(np.clip(blurred, 0, 1) * 255) / 255.0
B = np.fft.fft2(blurred)


# 1) Naive inverse filter: X = B / H
# H gets tiny at high frequencies, so dividing by it blows the rounding noise up
inverse = np.real(np.fft.ifft2(B / H))


# 2) Wiener filter: X = B * conj(H) / (|H|^2 + k)
# k stops the division from exploding where H is small (more k = less noise, less sharpness)
k = 0.001
wiener = np.real(np.fft.ifft2(B * np.conj(H) / (np.abs(H) ** 2 + k)))


# Print how close each result is to the original (PSNR in dB, higher = better)
def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))

print(f"Blurred:         {psnr(image, blurred):.2f} dB")
print(f"Inverse filter:  {psnr(image, np.clip(inverse, 0, 1)):.2f} dB")
print(f"Wiener filter:   {psnr(image, np.clip(wiener, 0, 1)):.2f} dB")


# Convert back to 8-bit for display
def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

cv2.imshow('Original Image', to_uint8(image))
cv2.imshow('Gaussian Blurred Image', to_uint8(blurred))
cv2.imshow('Inverse Filter (noise blows up)', to_uint8(inverse))
cv2.imshow('Wiener Filter', to_uint8(wiener))

cv2.waitKey(0)
cv2.destroyAllWindows()
