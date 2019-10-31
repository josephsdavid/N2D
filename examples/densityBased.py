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
x,y = data.load_mnist()

n_clusters = 10
mnistcluster = nd.n2d(x, nclust = n_clusters)

mnistcluster.preTrainEncoder(weights = "mnist-1000-ae_weights.h5")

import umap
import hdbscan


class UmapHDB:
   def __init__(self, min_samples, min_cluster_size,
                umapdim = 2,
                umapN = 10,
                umapMd = float(0),
                umapMetric = 'euclidean',
                random_state = 0):
       self.min_samples = min_samples
       self.min_cluster_size = min_cluster_size
  # change this bit for changing the manifold learner
       self.manifoldInEmbedding = umap.UMAP(
           random_state = random_state,
           metric = umapMetric,
           n_components = umapdim,
           n_neighbors = umapN,
           min_dist = umapMd
       )
  # change this bit to change the clustering mechanism
       self.clusterManifold = hdbscan.HDBSCAN(
   		min_samples = min_samples,
   		min_cluster_size = 500
   	)

       self.hle = None


   def predict(self, hl):
       # obviously if you change the clustering method or the manifold learner
       # youll want to change the predict method too.
       self.hle = self.manifoldInEmbedding.fit_transform(hl)
       y_pred = self.clusterManifold.fit_predict(self.hle)
       return(y_pred >= 0)


manifoldDB = UmapHDB(10, 500)

mnistcluster.predict(manifoldDB)


# Does not work
# mnistcluster.visualize(y, None, dataset = "mnist-hdb", nclust = 10)
print(mnistcluster.assess(y)) # we are not clustering much of the data again
