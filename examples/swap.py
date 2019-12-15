import n2d
from sklearn.cluster import KMeans
import umap

from n2d import datasets as data
x, y, y_names = data.load_har()


class UmapKmeans:
    # you can pass whatever parameters you need too.
    def __init__(self, nclust,
                 umapdim = 2,
                 umapN = 10,
                 umapMd = float(0),
                 umapMetric = 'euclidean',
                 random_state = 0
                 ):
        self.nclust = nclust
        # change this bit for changing the manifold learner
        self.manifoldInEmbedding = umap.UMAP(
            random_state = random_state,
            metric = umapMetric,
            n_components = umapdim,
            n_neighbors = umapN,
            min_dist = umapMd
        )
        # change this bit to change the clustering mechanism
        self.clusterManifold = KMeans(
            n_clusters = nclust,
            random_state = random_state,
            n_jobs = -1
        )

        self.hle = None

    def fit(self, hl):
        self.hle = self.manifoldInEmbedding.fit_transform(hl)
        self.clusterManifold.fit(self.hle)

    def predict(self):
        # obviously if you change the clustering method or the manifold learner
        # youll want to change the predict method too.
        y_pred = self.clusterManifold.predict(self.hle)
        return(y_pred)

    # transform new data into the manifold, and then predict that
    def transform(self, x):
        manifold = self.manifoldInEmbedding.transform(x)
        y_pred = self.clusterManifold.predict(manifold)
        return(np.asarray(y_pred))



manifoldKM = UmapKmeans(6)
kmclust = n2d.n2d(x, manifoldKM, ndim = 6)

# now we can continue as normal!

kmclust.fit(weights = "weights/har-1000-ae_weights.h5")

kmclust.predict()
print(kmclust.assess(y))

