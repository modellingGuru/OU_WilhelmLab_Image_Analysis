import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import plotly.graph_objects as go

# Load segmented nanoparticle coordinates 
nanoparticles = np.loadtxt("nanoparticle_coordinates.csv", delimiter=",")

# 1. Compute Nanoparticle Density
voxel_size = (0.2, 0.2, 0.2)  # µm per voxel (example)
volume_um = np.array(image_shape) * voxel_size
total_volume = np.prod(volume_um)  # µm³

density = len(nanoparticles) / total_volume  # Nanoparticles per unit volume

# 2. Compute Nearest-Neighbor Distances (For Cluster Analysis)
tree = KDTree(nanoparticles)
distances, _ = tree.query(nanoparticles, k=2)  # k=2 because first neighbor is itself
nearest_neighbor_distances = distances[:, 1]  # Ignore distance to self

# 3. Plot Nearest-Neighbor Distance Distribution
plt.figure(figsize=(6, 4))
sns.histplot(nearest_neighbor_distances, bins=30, kde=True, color="red")
plt.xlabel("Nearest Neighbor Distance (µm)")
plt.ylabel("Frequency")
plt.title("Nanoparticle Spatial Clustering")
plt.show()

# Save Metrics
np.savetxt("nearest_neighbor_distances.csv", nearest_neighbor_distances, header="distance_um", comments='')
with open("nanoparticle_metrics.txt", "w") as f:
    f.write(f"Total particles: {len(nanoparticles)}\n")
    f.write(f"Density: {density:.4f} particles/µm³\n")
    f.write(f"Mean NND: {np.mean(nearest_neighbor_distances):.2f} µm\n")
    f.write(f"Std NND: {np.std(nearest_neighbor_distances):.2f} µm\n")

#  Print  Metrics 
print(f"Total Nanoparticles: {len(nanoparticles)}")
print(f"Nanoparticle Density: {density:.2f} particles/µm³")
print(f"Mean Nearest Neighbor Distance: {np.mean(nearest_neighbor_distances):.2f} µm")
print(f"Standard Deviation of Nearest Neighbor Distances: {np.std(nearest_neighbor_distances):.2f} µm")
