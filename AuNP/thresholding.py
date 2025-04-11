import numpy as np
import dask.array as da
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import dilation, erosion, ball
from scipy import ndimage

def downsampling(image, max_size=512):
    """Automatically determines an optimal downsampling factor for large images."""
    factor = max(1, image.shape[0] // max_size, image.shape[1] // max_size, image.shape[2] // max_size)
    return factor

def make_3d_structured_element(radius, shape="sphere"):
    """Creates a 3D structured element for morphological operations."""
    if shape == "sphere":
        return ball(radius)
    raise ValueError("Unsupported shape type")

def apply_otsu_threshold(image, down_sampling=None, multi_otsu=False):
    """Applies Otsu or Multi-Otsu thresholding with adaptive sampling."""
    if down_sampling is None:
        down_sampling = adaptive_downsampling(image)
    
    blurred = da.map_blocks(ndimage.gaussian_filter, image, sigma=1, dtype=image.dtype)
    sampled_image = blurred[::down_sampling, ::down_sampling, ::down_sampling]
    
    if multi_otsu:
        thresholds = threshold_multiotsu(sampled_image.compute(), classes=3)
        image_binary = blurred.map_blocks(lambda block: (block > thresholds[1]), dtype=bool)
    else:
        threshold = threshold_otsu(sampled_image.compute())
        image_binary = blurred.map_blocks(lambda block: block > threshold, dtype=bool)
    
    return image_binary

def morphological_processing(image_binary, dilate_radius=2, erode_radius=2):
    """Applies dilation, hole-filling, and erosion to refine segmentation."""
    structured_element_dilate = make_3d_structured_element(dilate_radius)
    structured_element_erode = make_3d_structured_element(erode_radius)
    
    image_binary = image_binary.map_overlap(
        lambda block: dilation(block, structured_element_dilate), depth=10)
    image_binary = image_binary.map_overlap(
        lambda block: ndimage.binary_fill_holes(block), depth=10)
    image_binary = image_binary.map_overlap(
        lambda block: erosion(block, structured_element_erode), depth=10).astype(np.double)
    
    return image_binary

def otsu_segmentation_pipeline(image, multi_otsu=False, dilate_radius=2, erode_radius=2):
    """Full Otsu segmentation pipeline including morphological processing."""
    logging.info("Starting Otsu thresholding...")
    binary_image = apply_otsu_threshold(image, multi_otsu=multi_otsu)
    logging.info("Applying morphological processing...")
    processed_image = morphological_processing(binary_image, dilate_radius, erode_radius)
    logging.info("Segmentation complete.")
    return processed_image
