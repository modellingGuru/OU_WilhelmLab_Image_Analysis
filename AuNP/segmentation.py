import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label


def remove_close_blobs(blobs, image, min_dist):
    """
    Removes blobs that are too close to each other, keeping the highest intensity one.
    """
    if blobs.size == 0:
        return blobs

    coords = blobs[:, :image.ndim]
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
    """
    if blobs.size == 0:
        return blobs

    intensities = np.array([image[tuple(blob[:image.ndim])] for blob in blobs])
    sorted_indices = np.argsort(intensities)[::-1]
    return blobs[sorted_indices]


def eliminate_insignificant_blobs(image, blobs, sigma_threshold=3):
    """
    Removes blobs with intensity below a statistical significance threshold.
    """
    if blobs.size == 0:
        return blobs

    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    intensity_threshold = mean_intensity + sigma_threshold * std_intensity

    significant_blobs = [blob for blob in blobs if image[tuple(blob[:image.ndim])] >= intensity_threshold]
    return np.array(significant_blobs)


def apply_otsu_threshold(image):
    """
    Applies Otsu’s thresholding method to binarize the image.
    """
    from skimage.filters import threshold_otsu

    image_blurred = gaussian_filter(image, sigma=1)
    threshold = threshold_otsu(image_blurred)
    return image_blurred > threshold


def segment_nanoparticles(image, min_dist=2.0, sigma_threshold=2):
    """
    Complete nanoparticle segmentation pipeline for 2D or 3D images.
    """
    binary_image = apply_otsu_threshold(image)
    labeled, num_features = label(binary_image)

    # Extract blob coordinates
    blobs = np.argwhere(labeled > 0)

    # For 2D images, pad to 3D for consistency (x, y, z)
    if image.ndim == 2:
        blobs = np.column_stack([blobs, np.zeros(len(blobs), dtype=int)])

    blobs = sort_by_intensity(blobs, image)
    blobs = remove_close_blobs(blobs, image, min_dist)
    blobs = eliminate_insignificant_blobs(image, blobs, sigma_threshold)

    return blobs


def save_coordinates_to_csv(blobs, output_path="nanoparticles.csv"):
    """
    Save blob coordinates to CSV in x,y,z format.
    """
    if blobs.shape[1] >= 3:
        reordered = blobs[:, [2, 1, 0]]  # From z,y,x to x,y,z
    else:
        reordered = blobs
    np.savetxt(output_path, reordered, fmt='%d', delimiter=",", header="x,y,z", comments="")
    print(f"Saved {len(reordered)} blob coordinates to {output_path}")



