import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import plotly.graph_objects as go


def load_coordinates(file_path="nanoparticle_coordinates.csv"):
    """
    Loads nanoparticle coordinates from a CSV file.
    """
    try:
        coords = np.loadtxt(file_path, delimiter=",")
        if coords.ndim == 1:
            coords = np.expand_dims(coords, axis=0)
        return coords
    except Exception as e:
        raise RuntimeError(f"Failed to load coordinates: {e}")


def calculate_density(coords, image_shape, voxel_size):
    """
    Calculates nanoparticle density in 3D space.

    Args:
        coords (np.ndarray): Particle coordinates.
        image_shape (tuple): Image shape (z, y, x).
        voxel_size (tuple): Voxel size in µm.

    Returns:
        tuple: (particle count, volume µm³, density particles/µm³)
    """
    volume_um = np.array(image_shape) * np.array(voxel_size)
    total_volume = np.prod(volume_um)
    count = len(coords)
    density = count / total_volume if total_volume > 0 else 0
    return count, total_volume, density

def compute_nnd(coords):
    """
    Computes nearest neighbor distances from particle coordinates.

    Args:
        coords (np.ndarray): Particle coordinates.

    Returns:
        np.ndarray: Nearest-neighbor distances.
    """
    if len(coords) < 2:
        return np.array([])

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1]  # Skip distance to self


def plot_nnd_histogram(nnd_array, output_path=None, parent_widget=None):
    """
    Plots a histogram of nearest-neighbor distances.

    Args:
        nnd_array (np.ndarray): NND values.
        output_path (str): Optional file to save plot.
        parent_widget: Optional Tkinter widget to embed plot.
    """
    if len(nnd_array) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(nnd_array, bins=30, color="red", alpha=0.7)
    plt.xlabel("Nearest Neighbor Distance (µm)")
    plt.ylabel("Frequency")
    plt.title("Nanoparticle Spatial Clustering")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    elif parent_widget:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(plt.gcf(), master=parent_widget)
        canvas.draw()
        canvas.get_tk_widget().pack()
    else:
        plt.show()

    plt.close()


def plot_3d_scatter(coords, output_html="scatter3d.html"):
    """
    Plots 3D scatter of nanoparticle coordinates using Plotly.

    Args:
        coords (np.ndarray): Particle coordinates.
        output_html (str): Path to save HTML.

    Returns:
        str: HTML file path.
    """
    if coords.shape[1] < 3:
        coords = np.column_stack([coords, np.zeros(len(coords))])

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
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
    Saves NND and summary metrics to files.

    Args:
        coords (np.ndarray): Coordinates of particles.
        nnd_array (np.ndarray): NND distances.
        density (float): Density value.
        output_dir (str): Folder to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save distances
    nnd_file = os.path.join(output_dir, "nearest_neighbor_distances.csv")
    np.savetxt(nnd_file, nnd_array, delimiter=",", fmt="%.4f", header="distance_um", comments='')

    # Save summary
    metrics_file = os.path.join(output_dir, "nanoparticle_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Total particles: {len(coords)}\n")
        f.write(f"Density: {density:.4f} particles/µm³\n")
        f.write(f"Mean NND: {np.mean(nnd_array):.2f} µm\n")
        f.write(f"Std NND: {np.std(nnd_array):.2f} µm\n")
