# Third party modules
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans

# First party modules
import n2d
from n2d import datasets as data

x, y, y_names = data.load_har()


class UmapKmeans:
    # you can pass whatever parameters you need too.
    def __init__(
        self,
        n_clusters,
        umap_dim=2,
        umap_neighbors=10,
        umap_min_distance=float(0),
        umap_metric="euclidean",
        random_state=0,
    ):

        self.n_clusters = n_clusters

        self.manifold_in_embedding = umap.UMAP(
            random_state=random_state,
            metric=umap_metric,
            n_components=umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_distance,
        )

        self.cluster_manifold = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_jobs=-1
        )
        self.hle = None

    def fit(self, hl):
        self.hle = self.manifold_in_embedding.fit_transform(hl)
        self.cluster_manifold.fit(self.hle)

    def predict(self, hl):
        manifold = self.manifold_in_embedding.transform(hl)
        y_pred = self.cluster_manifold.predict(manifold)
        return np.asarray(y_pred)

    def fit_predict(self, hl):
        self.hle = self.manifold_in_embedding.fit_transform(hl)
        self.cluster_manifold.fit(self.hle)
        y_pred = self.cluster_manifold.predict(self.hle)
        return np.asarray(y_pred)


manifoldKM = UmapKmeans(6)
kmclust = n2d.n2d(x.shape[-1], manifoldKM, 6)

# now we can continue as normal!

kmclust.fit(x, weights="weights/har-1000-ae_weights.h5")

kmclust.predict(x)
print(kmclust.assess(y))
kmclust.visualize(y, None, 6)
plt.show()
