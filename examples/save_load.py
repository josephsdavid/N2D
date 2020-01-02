import numpy as np
import tensorflow as tf
import os
import random as rn
import n2d
from n2d import datasets as data


x, y = data.load_mnist()

n_clusters = 10


#manifoldGMM = n2d.UmapGMM(n_clusters, umap_neighbors=10)
#mnistcluster = n2d.n2d(x.shape[-1], manifoldGMM, n_clusters)
#
#
#mnistcluster.fit(x, pretrain_epochs=2)
#
#n2d.save_n2d(mnistcluster, 'test_ae.h5', 'test_man.sav')


mod = n2d.load_n2d('test_ae.h5', 'test_man.sav')

preds = mod.predict(x)

print(preds)

print(mod.assess(y))
