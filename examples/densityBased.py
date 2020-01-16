import os
import n2d
import random as rn
import numpy as np
import n2d.datasets as data
import hdbscan
import umap

# load up mnist example
x,y = data.load_mnist()

# autoencoder can be just passed normally, see the other examples for extending
# it
ae = n2d.AutoEncoder(input_dim=x.shape[-1], output_dim=20)

# arguments for clusterer go in a dict
hdbscan_args = {"min_samples":10,"min_cluster_size":500, 'prediction_data':True}

# arguments for manifold learner go in a dict
umap_args = {"metric":"euclidean", "n_components":2, "n_neighbors":30,"min_dist":0}

# pass the classes and dicts into the generator
# manifold class, manifold args, cluster class, cluster args
db = n2d.manifold_cluster_generator(umap.UMAP, umap_args, hdbscan.HDBSCAN, hdbscan_args)

# pass the manifold-cluster tool and the autoencoder into the n2d class
db_clust = n2d.n2d(db, ae)

# fit
db_clust.fit(x, epochs = 10)

# the clusterer is a normal hdbscan object
print(db_clust.clusterer.probabilities_)

print(db_clust.clusterer.labels_)

# access the manifold learner at
print(db_clust.manifolder)


# if the parent classes have a method you can likely use it (make an issue if not)
db_clust.fit_predict(x, epochs = 10)

# however this will error because hdbscan doesnt have that method
db_clust.predict(x)

# predict on new data with the approximate prediction

x_test, y_test = data.load_mnist_test()


# access the parts of the autoencoder within n2d or outside of it
test_embedding = ae.encoder.predict(x_test)
test_n2d_embedding = db_clust.encoder.predict(x_test)

test_embedding - test_n2d_embedding
# all zeros

test_labels, strengths = hdbscan.approximate_predict(db_clust.clusterer, db_clust.manifolder.transform(test_embedding))

print(test_labels)
print(strengths)
