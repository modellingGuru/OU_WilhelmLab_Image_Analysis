import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label


def remove_close_blobs(blobs, image, min_dist):
    """
    Removes blobs that are too close to each other, keeping the highest intensity one.

    Args:
        blobs (np.ndarray): Array of blob coordinates and optional metadata.
        image (np.ndarray): 3D image from which blobs are extracted.
        min_dist (float): Minimum allowed distance between blobs.

    Returns:
        np.ndarray: Filtered list of blob coordinates.
    """
    if blobs.size == 0:
        return blobs

    if blobs.shape[1] < 3:
        raise ValueError("Blobs array must have at least 3 columns for x, y, z coordinates.")

    coords = blobs[:, :3]
    tree = cKDTree(coords)
    keep = np.ones(len(blobs), dtype=bool)

    for i, coord in enumerate(coords):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coord, min_dist)
        for j in neighbors:
            if i != j and image[tuple(coords[j])] < image[tuple(coords[i])]:
                keep[j] = False

    return blobs[keep]


def sort_by_intensity(blobs, image):
    """
    Sorts detected blobs based on their intensity values in descending order.

    Args:
        blobs (np.ndarray): Array of detected blobs.
        image (np.ndarray): 3D image containing intensity values.

    Returns:
        np.ndarray: Sorted blob coordinates based on intensity.
    """
    if blobs.size == 0:
        return blobs

    intensities = np.array([image[tuple(blob[:3])] for blob in blobs])
    sorted_indices = np.argsort(intensities)[::-1]
    return blobs[sorted_indices]


def eliminate_insignificant_blobs(image, blobs, sigma_threshold=3):
    """
    Removes blobs with intensity below a statistical significance threshold.

    Args:
        image (np.ndarray): 3D image containing intensity values.
        blobs (np.ndarray): Detected blob coordinates.
        sigma_threshold (float): Multiplier for standard deviation threshold.

    Returns:
        np.ndarray: Filtered blob array.
    """
    if blobs.size == 0:
        return blobs

    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    intensity_threshold = mean_intensity + sigma_threshold * std_intensity

    significant_blobs = [blob for blob in blobs if image[tuple(blob[:3])] >= intensity_threshold]
    return np.array(significant_blobs)


def apply_otsu_threshold(image):
    """
    Applies Otsu’s thresholding method to binarize the image.

    Args:
        image (np.ndarray): 3D grayscale image.

    Returns:
        np.ndarray: Binary image after thresholding.
    """
    from skimage.filters import threshold_otsu

    image_blurred = gaussian_filter(image, sigma=1)
    threshold = threshold_otsu(image_blurred)
    return image_blurred > threshold


def segment_nanoparticles(image, min_dist=2.0, sigma_threshold=3):
    """
    Complete nanoparticle segmentation pipeline.

    Args:
        image (np.ndarray): 3D image data.
        min_dist (float): Minimum separation distance for blobs.
        sigma_threshold (float): Intensity threshold multiplier.

    Returns:
        np.ndarray: Segmented and filtered nanoparticle coordinates.
    """
    binary_image = apply_otsu_threshold(image)
    labeled, num_features = label(binary_image)

    # Extract blob coordinates
    blobs = np.array(np.argwhere(labeled > 0))

    # Sort, filter close blobs, and eliminate insignificant ones
    blobs = sort_by_intensity(blobs, image)
    blobs = remove_close_blobs(blobs, image, min_dist)
    blobs = eliminate_insignificant_blobs(image, blobs, sigma_threshold)

    return blobs

def save_coordinates_to_csv(blobs, output_path="nanoparticles.csv"):
    np.savetxt(output_path, blobs, fmt='%d', delimiter=",", header="z,y,x", comments="")

