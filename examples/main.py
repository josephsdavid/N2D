import os
import n2d as nd
import random as rn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
matplotlib.use('agg')

import tensorflow as tf
from keras import backend as K



import datasets as data
x,y, y_names = data.load_har()

n_clusters = 6
harcluster = nd.n2d(x, nclust = n_clusters)

harcluster.preTrainEncoder(weights = "har-1000-ae_weights.h5")

manifoldGMM = nd.UmapGMM(n_clusters)

harcluster.predict(manifoldGMM)

harcluster.visualize(y, y_names, dataset = "har", nclust = n_clusters)
print(harcluster.assess(y))


from sklearn.cluster import SpectralClustering
import umap
class UmapSpectral:
    def __init__(self, nclust,
                 umapdim = 2,
                 umapN = 10,
                 umapMd = float(0),
                 umapMetric = 'euclidean',
                 random_state = 0):
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
        self.clusterManifold = SpectralClustering(
    		n_clusters = nclust,
    		affinity = 'nearest_neighbors',
    		random_state = random_state
    	)

        self.hle = None


    def predict(self, hl):
        # obviously if you change the clustering method or the manifold learner
        # youll want to change the predict method too.
        self.hle = self.manifoldInEmbedding.fit_transform(hl)
        self.clusterManifold.fit(self.hle)
        y_pred = self.clusterManifold.fit_predict(self.hle)
        return(y_pred)

manifoldSC = UmapSpectral(6)
harcluster.predict(manifoldSC)
print(harcluster.assess(y))
