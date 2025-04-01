import cv2
import numpy as np
from skimage.morphology import ball, disk
from skimage.transform import resize
from config import GAUSSIAN_SIGMA

def calculate_kernel_size(sigma):
    """Calculate the Gaussian kernel size ensuring it is an odd integer."""
    return int(6 * sigma + 1) | 1  

def apply_gaussian_blur(image, sigma=GAUSSIAN_SIGMA):
    """Apply Gaussian blur to a 2D or 3D image."""
    kernel_size = calculate_kernel_size(sigma)

    if image.ndim == 2:  # 2D Image
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    elif image.ndim == 3:  # 3D Image
        blurred = np.zeros_like(image)
        for i in range(image.shape[0]):  # Apply blur slice by slice
            blurred[i] = cv2.GaussianBlur(image[i], (kernel_size, kernel_size), sigmaX=sigma)
        return blurred
    else:
        raise ValueError("Image should be either 2D or 3D")

def make_3d_structured_element(radius, shape="sphere"):
    """Generate a 3D structured element for morphological operations."""
    if shape == "sphere":
        return ball(radius)
    elif shape == "disk":
        return disk(radius)
    else:
        raise ValueError("Invalid shape. Choose 'sphere' or 'disk'.")
