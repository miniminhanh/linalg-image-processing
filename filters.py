"""
The filters from this repo, wrapped as functions so the UI (app.py) can use them.

Each filter takes a BGR image (as returned by cv2.imread / cv2.imdecode) and returns
a list of (title, image) pairs - one pair per window the original script opens.
Filters that use a Gaussian kernel also take kernel_size and sigma, which the UI sets
with sliders; their defaults match the standalone scripts.
The maths matches the standalone scripts:

    basic-gaussian-blur.py   -> gaussian_blur
    basic-sharpening.py      -> sharpen
    basic-edge-detection.py  -> edge_detection
    basic-deblurring.py      -> deblur

To add a new filter to the UI, write a function here and add it to FILTERS at the bottom.
"""
import cv2
import numpy as np


def gaussian_blur(image, kernel_size=3, sigma=0.8):
    # 0.8 is the sigma OpenCV picks automatically for a 3x3 kernel (sigma=0 in basic-gaussian-blur.py)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return [('Gaussian Blurred Image', blurred_image)]


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return [('Sharpened Image', sharpened_image)]


def edge_detection(image, kernel_size=3, sigma=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # kernel_size and sigma control the smoothing applied before finding edges
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # Sobel kernels; the y kernel is the transpose of the x kernel
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
    sobel_y_kernel = sobel_x_kernel.T

    grad_x = cv2.filter2D(blurred, cv2.CV_64F, sobel_x_kernel)
    grad_y = cv2.filter2D(blurred, cv2.CV_64F, sobel_y_kernel)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    sobel_edges = cv2.convertScaleAbs(magnitude / max(magnitude.max(), 1e-12) * 255)

    canny_edges = cv2.Canny(blurred, 100, 200)

    return [('Sobel Edges', sobel_edges), ('Canny Edges', canny_edges)]


def deblur(image, kernel_size=7, sigma=2.0):
    # Same demo as basic-deblurring.py, but in colour: blur with a known Gaussian kernel,
    # then try to undo it. Blurring is linear and acts on each colour channel independently,
    # so each of the B, G, R channels goes through the same steps with the same kernel.
    shape = image.shape[:2]

    g = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = g @ g.T

    # The kernel's frequency response H only depends on the kernel, so compute it once.
    # rfft2 is the FFT for real-valued input: it skips the mirrored half of the spectrum,
    # which halves memory use without changing the result.
    padded_kernel = np.zeros(shape)
    padded_kernel[:kernel_size, :kernel_size] = kernel
    padded_kernel = np.roll(padded_kernel, (-(kernel_size // 2), -(kernel_size // 2)), axis=(0, 1))
    H = np.fft.rfft2(padded_kernel)

    k = 0.001  # Wiener regularisation (more k = less noise, less sharpness)
    wiener_filter = np.conj(H) / (np.abs(H) ** 2 + k)

    def to_uint8(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    names = ('Gaussian Blurred', 'Inverse Filter', 'Wiener Filter')
    outputs = {name: np.empty_like(image) for name in names}
    squared_error = dict.fromkeys(names, 0.0)

    # One channel at a time keeps memory low on small servers
    for c in range(image.shape[2]):
        original = image[:, :, c].astype(np.float64) / 255.0

        blurred = np.fft.irfft2(np.fft.rfft2(original) * H, s=shape)
        # Save to 8-bit like a real image file would; this rounding is the only "noise" we add
        blurred = np.round(np.clip(blurred, 0, 1) * 255) / 255.0
        B = np.fft.rfft2(blurred)

        results = {
            'Gaussian Blurred': blurred,
            'Inverse Filter': np.fft.irfft2(B / H, s=shape),            # naive inverse filter
            'Wiener Filter': np.fft.irfft2(B * wiener_filter, s=shape),  # Wiener filter
        }
        for name, result in results.items():
            result = np.clip(result, 0, 1)
            squared_error[name] += np.sum((original - result) ** 2)
            outputs[name][:, :, c] = to_uint8(result)

    def psnr(name):
        # PSNR from the mean squared error over all three channels, as a label
        mse = squared_error[name] / image.size
        # Below 1e-10 the difference is just floating-point rounding, far smaller than one 8-bit step
        return 'identical' if mse < 1e-10 else f'{10 * np.log10(1.0 / mse):.1f} dB'

    return [(f'{name} ({psnr(name)})', outputs[name]) for name in names]


# Slider options shared by every filter that uses a Gaussian kernel
KERNEL_SIZES = (3, 5, 7)
SIGMA_MIN, SIGMA_MAX = 0.01, 3.0

# id -> display name, description, function, and default (kernel_size, sigma)
# (None means the filter has no Gaussian kernel, so the UI hides the sliders)
FILTERS = {
    'gaussian-blur': {
        'name': 'Gaussian Blur',
        'description': 'Smooths the image with a Gaussian kernel.',
        'func': gaussian_blur,
        'gaussian_defaults': (3, 0.8),
    },
    'sharpening': {
        'name': 'Sharpening',
        'description': '3x3 sharpening kernel [[0,-1,0],[-1,5,-1],[0,-1,0]].',
        'func': sharpen,
        'gaussian_defaults': None,
    },
    'edge-detection': {
        'name': 'Edge Detection',
        'description': 'Gaussian smoothing, then Sobel gradient magnitude and Canny edges (thresholds 100/200).',
        'func': edge_detection,
        'gaussian_defaults': (3, 0.8),
    },
    'deblurring': {
        'name': 'Deblurring Demo',
        'description': 'Blurs the image with a known Gaussian kernel, then undoes it with a naive inverse '
                       'filter and a Wiener filter. PSNR vs. the original is shown in each title.',
        'func': deblur,
        'gaussian_defaults': (7, 2.0),
    },
}
