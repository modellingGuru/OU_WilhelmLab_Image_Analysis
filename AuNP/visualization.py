import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter    
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_localization_data(csv_path=None):
    """
    Loads CSV containing distance-to-surface and neighbor-distance metrics.

    Args:
        csv_path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with expected columns.
    """
    if csv_path is None:
        csv_path = os.path.expanduser("nanoparticle_coordinates.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def plot_localization_kde(df, output_path="cellular_localization_kde.png", return_fig=False):
    """
    Generates and optionally saves a KDE scatter visualization for nanoparticle localization.

    Args:
        df (pd.DataFrame): DataFrame with required columns.
        output_path (str): Path to save the output image.
        return_fig (bool): If True, return the matplotlib figure object for embedding.

    Returns:
        matplotlib.figure.Figure (optional): Only returned if return_fig=True.
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_axisbelow(True)

    # Define axis bounds
    x_min, x_max = -50, 50
    y_min, y_max = 10, 50

    # KDE Plot
    kde = sns.kdeplot(
        data=df,
        x="Shortest Distance to Surfaces Surfaces=Surfaces 1",
        y="Average Distance To 5 Nearest Neighbours",
        fill=True,
        cmap="magma",
        levels=20,
        alpha=0.8,
        cbar=True,
        cbar_kws={'label': 'Density'}
    )

    # Labels and grid
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(x_min * 0.7, y_max * 0.95, "Intracellular",
             fontsize=14, ha='center', color='darkblue', backgroundcolor='white', alpha=0.7)
    plt.text(x_max * 0.7, y_max * 0.95, "Extracellular",
             fontsize=14, ha='center', color='darkblue', backgroundcolor='white', alpha=0.7)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title('Nanoparticle Localization and Density', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Shortest Distance to Cell Surface', fontsize=14, labelpad=10)
    plt.ylabel('Average Distance To 5 Nearest Neighbours', fontsize=14, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Customize colorbar
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Density', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if return_fig:
        return fig
    else:
        plt.close(fig)


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

    # return plot
    return fig


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

    return fig

# Save visual 
def save_data(image, blobs, image_path="synthetic_image.npy", csv_path="nanoparticles.csv"):
    np.save(image_path, image)
    np.savetxt(csv_path, blobs, delimiter=",", header="x,y,z", comments='', fmt='%d')
    print(f"Saved image to {image_path}")
    print(f"Saved blob coordinates to {csv_path}")
