import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import plotly.graph_objects as go


def load_coordinates(file_path="nanoparticle_coordinates.csv"):
    """
    Loads nanoparticle coordinates from a CSV file.
    """
    try:
        return np.loadtxt(file_path, delimiter=",")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found. Make sure the CSV exists.")


def calculate_density(coords, image_shape, voxel_size):
    """
    Calculates density of nanoparticles in a volume.

    Args:
        coords (np.ndarray): Coordinates of nanoparticles.
        image_shape (tuple): Shape of the image (z, y, x or similar).
        voxel_size (tuple): Voxel size in micrometers.

    Returns:
        count (int), volume (float), density (float)
    """
    volume_um = np.array(image_shape) * np.array(voxel_size)
    total_volume = np.prod(volume_um)  # in µm³
    density = len(coords) / total_volume if total_volume > 0 else 0
    return len(coords), total_volume, density


def compute_nnd(coords):
    """
    Computes nearest neighbor distances for the given coordinates.

    Args:
        coords (np.ndarray): Particle coordinates.

    Returns:
        np.ndarray: Array of nearest-neighbor distances.
    """
    tree = KDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return distances[:, 1]  # Exclude self-distance


def plot_nnd_histogram(nnd_array, output_path=None):
    """
    Plots a histogram of nearest-neighbor distances.

    Args:
        nnd_array (np.ndarray): Array of nearest-neighbor distances.
        output_path (str): If provided, saves the plot to file.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(nnd_array, bins=30, color="red", alpha=0.7)
    plt.xlabel("Nearest Neighbor Distance (µm)")
    plt.ylabel("Frequency")
    plt.title("Nanoparticle Spatial Clustering")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def plot_3d_scatter(coords, output_html="scatter3d.html"):
    """
    Creates an interactive 3D scatter plot using Plotly.

    Args:
        coords (np.ndarray): Coordinates of particles.
        output_html (str): File path to save the HTML plot.

    Returns:
        str: Path to saved HTML file.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(
        title="3D Scatter Plot of Nanoparticles",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.write_html(output_html)
    return output_html


def save_metrics(coords, nnd_array, density, output_dir="."):
    """
    Saves metrics to text and CSV files.

    Args:
        coords (np.ndarray): Particle coordinates.
        nnd_array (np.ndarray): Nearest-neighbor distances.
        density (float): Calculated density.
        output_dir (str): Directory to save results.
    """
    np.savetxt(f"{output_dir}/nearest_neighbor_distances.csv", nnd_array,
               header="distance_um", comments='')

    with open(f"{output_dir}/nanoparticle_metrics.txt", "w") as f:
        f.write(f"Total particles: {len(coords)}\n")
        f.write(f"Density: {density:.4f} particles/µm³\n")
        f.write(f"Mean NND: {np.mean(nnd_array):.2f} µm\n")
        f.write(f"Std NND: {np.std(nnd_array):.2f} µm\n")
