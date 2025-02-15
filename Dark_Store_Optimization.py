import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

# Step 1: Apply Huff’s Gravity Model for Demand Capture
def huff_gravity_model(demand, stores, attractiveness, lambda_factor=2):
    probabilities = []
    for d in demand:
        total = sum(attractiveness[i] / (np.linalg.norm(d - stores[i]) ** lambda_factor)
                    for i in range(len(stores)))
        probabilities.append([attractiveness[i] / (np.linalg.norm(d - stores[i]) ** lambda_factor) / total
                              for i in range(len(stores))])
    return np.array(probabilities)

# Simulated Data
np.random.seed(42)
demand_points = np.random.rand(10, 2) * 10  # 10 demand locations
store_candidates = np.random.rand(5, 2) * 10  # 5 potential store locations
store_attractiveness = np.random.rand(5) * 100  # Attractiveness of each store

# Compute demand capture probabilities
huff_probabilities = huff_gravity_model(demand_points, store_candidates, store_attractiveness)

# Step 2: Use K-Means Clustering to Identify Initial Dark Store Placement
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(demand_points)
dark_store_initial_centers = kmeans.cluster_centers_

# Step 3: Use K-Medoids for Dark Store Refinement
kmedoids = KMedoids(n_clusters=3, random_state=42, metric="manhattan").fit(dark_store_initial_centers)
dark_store_centers = kmedoids.cluster_centers_

# Step 4: Generate Voronoi Diagram for Dark Store Zones
vor = Voronoi(dark_store_centers)
fig, ax = plt.subplots(figsize=(10, 10))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1)
ax.scatter(demand_points[:, 0], demand_points[:, 1], c='red', label="Demand Points", s=50)
ax.scatter(dark_store_centers[:, 0], dark_store_centers[:, 1], c='green', marker='X', label="Refined Dark Stores", s=200)
plt.title("Optimized Dark Store Placement with Refinement")
plt.legend()
plt.show()
