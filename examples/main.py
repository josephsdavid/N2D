# example


# Core Library modules
import os
import random as rn

# Third party modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.utils import print_summary

# First party modules
import n2d
from n2d import datasets as data

plt.style.use(["seaborn-white", "seaborn-paper"])
sns.set_context("paper", font_scale=1.3)


os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

rn.seed(0)

tf.set_random_seed(0)

np.random.seed(0)


# x,y = data.load_pendigits()
#
# n_clusters = 10
# pencluster = n2d.n2d(x, nclust = n_clusters, ae_args = {"act":"relu"})
#
# pencluster.preTrainEncoder(weight_id="pendigits")
#
# manifoldGMM = n2d.UmapGMM(n_clusters, umapdim=n_clusters)
#
# pencluster.predict(manifoldGMM)
#
##pencluster.visualize(y, names=None, dataset = "pendigits", nclust = n_clusters)
# print(pencluster.assess(y))

x, y = data.load_mnist()

n_clusters = 10
manifoldGMM = n2d.UmapGMM(n_clusters, umapN=10)
mnistcluster = n2d.n2d(x, manifoldGMM, ndim=n_clusters)


print_summary(mnistcluster.autoencoder.Model)


mnistcluster.fit(weights="weights/mnist-1000-ae_weights.h5", patience=None)

preds = mnistcluster.predict()
mnistcluster.visualize(y, None, nclust=n_clusters)
plt.show()

print(mnistcluster.assess(y))

x_test, y_test = data.load_mnist_test()

# assign new variables in same embedding, transform using autoencoder -> put in
# same UMAP embedding as the training data -> predict clusters using GMM trained
# on the training set
trans = mnistcluster.transform(x_test)
mnistcluster.visualize(y_test, None, nclust=n_clusters)
plt.show()

print(mnistcluster.assess(y_test))

trans

preds = mnistcluster.predict()
mnistcluster.visualize(y, None, nclust=n_clusters)
plt.show()

print(mnistcluster.assess(y))


# from sklearn.cluster import SpectralClustering
# import umap
# class UmapSpectral:
#    def __init__(self, nclust,
#                 umapdim = 2,
#                 umapN = 10,
#                 umapMd = float(0),
#                 umapMetric = 'euclidean', random_state = 0
#                 ):
#        self.nclust = nclust
# 	# change this bit for changing the manifold learner
#        self.manifoldInEmbedding = umap.UMAP(
#            random_state = random_state,
#            metric = umapMetric,
#            n_components = umapdim,
#            n_neighbors = umapN,
#            min_dist = umapMd
#        )
# 	# change this bit to change the clustering mechanism
#        self.clusterManifold = SpectralClustering(n_clusters = nclust, affinity = 'nearest_neighbors',random_state = random_state)
#
#        self.hle = None
#
#    def fit(self, hl):
#        self.hle = self.manifoldInEmbedding.fit_transform(hl)
#        self.clusterManifold.fit(self.hle)
#
#    def predict(self):
#    # obviously if you change the clustering method or the manifold learner
#    # youll want to change the predict method too.
#        y_pred = self.clusterManifold.fit_predict(self.hle)
#        return(y_pred)
#
##f_x, f_y, f_names = data.load_fashion()
##
##n_cl_fashion = 10
##
##fashioncl = n2d.n2d(f_x, nclust = n_cl_fashion)
##fashioncl.preTrainEncoder(weights = "fashion-1000-ae_weights.h5")
##
##manifold_cluster = n2d.UmapGMM(n_cl_fashion)
##fashioncl.predict(manifold_cluster)
##
##fashioncl.visualize(f_y, f_names, dataset = "fashion", nclust = n_cl_fashion)
##print(fashioncl.assess(f_y))
#
#
#
# manifoldSC = UmapSpectral(n_clusters)
# SCclust = n2d.n2d(x, manifoldSC, ndim = n_clusters)
## weights from the examples folder
# SCclust.fit(weights = "weights/har-1000-ae_weights.h5")
# SCclust.predict()
# print(SCclust.assess(y))

# m_x, m_y = data.load_mnist()
#
# n_cl_mnist = 10
#
# mnistcl = n2d.n2d(m_x, nclust = n_cl_mnist)
# mnistcl.preTrainEncoder(weights="mnist-1000-ae_weights.h5")
#
# mnistManifold = n2d.UmapGMM(n_cl_mnist)
# mnistcl.predict(mnistManifold)
#
# mnistcl.visualize(m_y, names = None, dataset = "mnist", nclust = n_cl_mnist)
# print(mnistcl.assess(m_y))
#
# from sklearn.cluster import SpectralClustering
# import umap
##class UmapSpectral:
#    def __init__(self, nclust,
#                 umapdim = 2,
#                 umapN = 10,
#                 umapMd = float(0),
#                 umapMetric = 'euclidean',
#                 random_state = 0):
#        self.nclust = nclust
# 	# change this bit for changing the manifold learner
#        self.manifoldInEmbedding = umap.UMAP(
#            random_state = random_state,
#            metric = umapMetric,
#            n_components = umapdim,
#            n_neighbors = umapN,
#            min_dist = umapMd
#        )
# 	# change this bit to change the clustering mechanism
#        self.clusterManifold = SpectralClustering(
#    		n_clusters = nclust,
#    		affinity = 'nearest_neighbors',
#    		random_state = random_state
#    	)
#
#        self.hle = None
#
#
#    def predict(self, hl):
#        # obviously if you change the clustering method or the manifold learner
#        # youll want to change the predict method too.
#        self.hle = self.manifoldInEmbedding.fit_transform(hl)
#        self.clusterManifold.fit(self.hle)
#        y_pred = self.clusterManifold.fit_predict(self.hle)
#        return(y_pred)
#
## manifoldSC = UmapSpectral(6)
## harcluster.predict(manifoldSC)
## print(harcluster.assess(y))
