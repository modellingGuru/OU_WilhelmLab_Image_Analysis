# Standard library imports
import os

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Expand user directory if needed and ensure the file path is absolute
file_directory = os.path.expanduser("~/Downloads/Scatter_Plot_1D.csv")

# Load the CSV file
df = pd.read_csv(file_directory)

# Display basic information about the dataframe
print(df.head())
print(df.info())
# Create a scatter plot with the specified columns
plt.figure(figsize=(10, 6))

# Create the scatter plot
sns.scatterplot(
    data=df,
    x="Shortest Distance to Surfaces Surfaces=Surfaces 1",
    y="Average Distance To 5 Nearest Neighbours",
    alpha=0.7,
    s=50  # Point size
)

# Enhance the plot
plt.title('Distance to Neighbors vs Distance to Surface')
plt.xlabel('Shortest Distance to Surface')
plt.ylabel('Average Distance To 5 Nearest Neighbours')
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

# Set the axes frame and ticks to be behind the plot content
ax.set_axisbelow(True)

# Set min and max values for limits
x_min = -50
x_max = 50
y_min = 10
y_max = 50

# Create the KDE plot with a nice colormap
kde = sns.kdeplot(
    data=df,
    x="Shortest Distance to Surfaces Surfaces=Surfaces 1",
    y="Average Distance To 5 Nearest Neighbours",
    fill=True,
    cmap="magma", #"viridis",
    levels=20,
    alpha=0.8,
    cbar=True,
    cbar_kws={'label': 'Density'}
)

# Add vertical line at x=0
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Add region labels
plt.text(x_min*0.7, y_max*0.95, "Intracellular",
         fontsize=14, ha='center', color='darkblue', backgroundcolor='white', alpha=0.7)
plt.text(x_max*0.7, y_max*0.95, "Extracellular",
         fontsize=14, ha='center', color='darkblue', backgroundcolor='white', alpha=0.7)

# Set limits based on data
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add title and labels with professional font sizes
plt.title('Nanoparticle Localization and Density', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Shortest Distance to Cell Surface', fontsize=14, labelpad=10)
plt.ylabel('Average Distance To 5 Nearest Neighbours', fontsize=14, labelpad=10)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Adjust colorbar
cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Density', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()

# Save high-resolution figure
plt.savefig('cellular_localization_kde.png', dpi=300, bbox_inches='tight')
