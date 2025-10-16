import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load saved feature data
data = np.load("kmeans_debug_feats.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

print(f"Loaded {len(X)} features.")

# Reduce 4D (L, a, b, S) to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Apply KMeans (2 clusters for Team A / Team B)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
centers_2d = pca.transform(kmeans.cluster_centers_)

# Scatter plot
plt.figure(figsize=(7,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=kmeans.labels_, cmap="coolwarm", s=25, alpha=0.7, label="Jersey features")
plt.scatter(centers_2d[:,0], centers_2d[:,1], c='black', s=150, marker='x', label='Cluster centers')

plt.title("KMeans Clustering of Jersey Colors (PCA 2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_visualization.png", dpi=200)
plt.show()
print("✅ Saved plot as kmeans_visualization.png")
