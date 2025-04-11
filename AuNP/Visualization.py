import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter

# Assuming the 'blobs' array contains [x, y, z] coordinates of detected nanoparticles
# and the intensity values from the 3D image

def viz_np(image, blobs, color_by_intensity=True):
    """
    Visualizes nanoparticles in 3D, displaying them with colors based on their intensity values.
    """
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates of detected nanoparticles
    x_vals = blobs[:, 0]
    y_vals = blobs[:, 1]
    z_vals = blobs[:, 2]

    # If color by intensity is selected, retrieve intensity at each particle position
    if color_by_intensity:
        intensities = np.array([image[x, y, z] for x, y, z in blobs])
        # Normalize intensity values to the range [0, 1]
        norm = plt.Normalize(vmin=np.min(intensities), vmax=np.max(intensities))
        colors = cm.viridis(norm(intensities))  # Color map can be changed
    else:
        # If no intensity coloring, default to a fixed color
        colors = np.array([0.5, 0.5, 0.5, 1.0])  # Gray color

    # Plot the nanoparticles as a 3D scatter plot
    ax.scatter(x_vals, y_vals, z_vals, c=colors, marker='o', s=10, alpha=0.7)

    # Add axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Slice Index)')
    ax.set_title('3D Visualization of Nanoparticles')

    # Show color bar if intensity-based coloring is applied
    if color_by_intensity:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax)
        cbar.set_label('Intensity')

    # Show plot
    plt.show()


def generate_mask(image, blobs, mask_size=3):
    """
    Generates a 3D binary mask based on the positions of nanoparticles.

    Parameters:
    -----------
    image: np.ndarray
        The 3D image where nanoparticles are detected.
    blobs: np.ndarray
        The coordinates of the detected nanoparticles.
    mask_size : int
        The size of the mask to be used for each nanoparticle (assumed spherical).

    Returns:
    --------
    np.ndarray: Mask image with nanoparticles marked.
    """
    mask = np.zeros_like(image, dtype=bool)
    
    for blob in blobs:
        x, y, z = [int(round(coord)) for coord in blob]  
        # Create a spherical mask around each nanoparticle
        x_range = np.arange(max(0, x - mask_size), min(image.shape[0], x + mask_size + 1))
        y_range = np.arange(max(0, y - mask_size), min(image.shape[1], y + mask_size + 1))
        z_range = np.arange(max(0, z - mask_size), min(image.shape[2], z + mask_size + 1))

        for xi in x_range:
            for yi in y_range:
                for zi in z_range:
                    # Only mask voxels within a spherical radius
                    if np.sqrt((xi - x)**2 + (yi - y)**2 + (zi - z)**2) <= mask_size:
                        mask[xi, yi, zi] = True

    return mask


def apply_mask(image, blobs, mask_size=3):
    """
    Applies the nanoparticle mask to the 3D image and visualizes it.

    Parameters:
    -----------
    image - np.ndarray
        The 3D image data.
    blobs - np.ndarray
        Coordinates of the detected nanoparticles.
    mask_size : int
        Size of the mask around each detected nanoparticle.
    """
    # Generate a mask based on the nanoparticles' coordinates
    nanoparticle_mask = generate_mask(image, blobs, mask_size)

    # Visualize the mask in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Coordinates of the mask's non-zero elements
    mask_coords = np.array(np.where(nanoparticle_mask))

    # Plotting the mask points in 3D
    ax.scatter(mask_coords[0], mask_coords[1], mask_coords[2], c='b', marker='o', alpha=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Slice Index)')
    ax.set_title('3D Mask Visualization of Nanoparticles')

    plt.show()


