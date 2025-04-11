import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter


# Load coordinates
def load_coordinates(csv_path, reorder_axes=False):
    """
    Loads blob coordinates from ("nanoparticles.csv") file.
    
    Redefine pathway to .csv file 
    """
    coords = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if reorder_axes:
        coords = coords[:, [2, 1, 0]]  # If needed
    return coords

# Generate Test Data
def generate_test_data(image_shape=(100, 100, 100), num_blobs=500, intensity_range=(100, 255)):
    """
    Generates a synthetic 3D image with random blobs for testing visualization.
    """
    image = np.zeros(image_shape, dtype=np.float32)
    blobs = []

    for _ in range(num_blobs):
        x = np.random.randint(0, image_shape[0])
        y = np.random.randint(0, image_shape[1])
        z = np.random.randint(0, image_shape[2])
        intensity = np.random.uniform(*intensity_range)

        image[x, y, z] = intensity
        blobs.append([x, y, z])

    image = gaussian_filter(image, sigma=1)
    return image, np.array(blobs)


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


def apply_mask(image, blobs, mask_size=3, colormap='viridis'):
    """
    Applies the nanoparticle mask to the 3D image using generate_mask,
    and visualizes it in 3D using color intensity mapping from the image.
    """
    mask = generate_mask(image, blobs, mask_size=mask_size)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Get intensities at blob centers
    intensities = np.array([image[int(x), int(y), int(z)] for x, y, z in blobs])
    norm = plt.Normalize(vmin=np.min(intensities), vmax=np.max(intensities))
    cmap = plt.cm.get_cmap(colormap)
    blob_colors = cmap(norm(intensities))

    for i, blob in enumerate(blobs):
        color = blob_colors[i]
        x, y, z = [int(round(coord)) for coord in blob]

        x_range = np.arange(max(0, x - mask_size), min(image.shape[0], x + mask_size + 1))
        y_range = np.arange(max(0, y - mask_size), min(image.shape[1], y + mask_size + 1))
        z_range = np.arange(max(0, z - mask_size), min(image.shape[2], z + mask_size + 1))

        coords = []

        for xi in x_range:
            for yi in y_range:
                for zi in z_range:
                    if mask[xi, yi, zi] and np.sqrt((xi - x)**2 + (yi - y)**2 + (zi - z)**2) <= mask_size:
                        coords.append([xi, yi, zi])

        coords = np.array(coords)
        if coords.size > 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       color=color, s=8, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Slice Index)')
    ax.set_title('3D Mask Visualization Colored by Intensity')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Intensity')

    plt.show()

# Save visual 
def save_data(image, blobs, image_path="synthetic_image.npy", csv_path="nanoparticles.csv"):
    np.save(image_path, image)
    np.savetxt(csv_path, blobs, delimiter=",", header="x,y,z", comments='', fmt='%d')
    print(f"Saved image to {image_path}")
    print(f"Saved blob coordinates to {csv_path}")
