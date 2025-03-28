import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Load segmented nanoparticle coordinates (x, y, z)
nanoparticles = np.load("nanoparticle_coordinates.npy")  # Assuming saved coordinates

# 1. Compute Nanoparticle Density
volume_size = (100, 100, 100)  # Replace with actual volume size (µm³ or voxel size)
total_volume = np.prod(volume_size)
density = len(nanoparticles) / total_volume  # Nanoparticles per unit volume

# 2. Compute Nearest-Neighbor Distances (For Cluster Analysis)
tree = KDTree(nanoparticles)
distances, _ = tree.query(nanoparticles, k=2)  # k=2 because first neighbor is itself
nearest_neighbor_distances = distances[:, 1]  # Ignore distance to self

# 3. Create 3D Scatter Plot of Nanoparticles
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(nanoparticles[:, 0], nanoparticles[:, 1], nanoparticles[:, 2], c="blue", s=5, alpha=0.6)
ax.set_xlabel("X Position (µm)")
ax.set_ylabel("Y Position (µm)")
ax.set_zlabel("Z Position (µm)")
ax.set_title("3D Distribution of Nanoparticles")
plt.show()

# 4. Plot Nearest-Neighbor Distance Distribution
plt.figure(figsize=(6, 4))
sns.histplot(nearest_neighbor_distances, bins=30, kde=True, color="red")
plt.xlabel("Nearest Neighbor Distance (µm)")
plt.ylabel("Frequency")
plt.title("Nanoparticle Spatial Clustering")
plt.show()

# 5. 3D Interactive Visualization with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=nanoparticles[:, 0], y=nanoparticles[:, 1], z=nanoparticles[:, 2],
    mode="markers",
    marker=dict(size=3, color=nanoparticles[:, 2], colorscale="Viridis", opacity=0.8)
)])
fig.update_layout(title="Interactive 3D View of Nanoparticles", scene=dict(
    xaxis_title="X Position", yaxis_title="Y Position", zaxis_title="Z Position"
))
fig.show()

# 6. Print Key Metrics for Academic Paper
print(f"Total Nanoparticles: {len(nanoparticles)}")
print(f"Nanoparticle Density: {density:.2f} particles/µm³")
print(f"Mean Nearest Neighbor Distance: {np.mean(nearest_neighbor_distances):.2f} µm")
print(f"Standard Deviation of Nearest Neighbor Distances: {np.std(nearest_neighbor_distances):.2f} µm")
