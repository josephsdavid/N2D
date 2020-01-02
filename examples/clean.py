import numpy as np
import tensorflow as tf
import os
import random as rn
import n2d
from n2d import datasets as data
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
rn.seed(0)
tf.set_random_seed(0)
np.random.seed(0)


x, y = data.load_mnist()

n_clusters = 10

manifoldGMM = n2d.UmapGMM(n_clusters, umap_neighbors=10)
mnistcluster = n2d.n2d(x.shape[-1], manifoldGMM, n_clusters)

# fit
mnistcluster.fit(x, weights="weights/mnist-1000-ae_weights.h5", patience=None)
preds_0 = mnistcluster.predict(x)

# fit_predict
preds_1 = mnistcluster.fit_predict(x,
                                   weights="weights/mnist-1000-ae_weights.h5",
                                   patience=None)
mnistcluster.visualize(y, None, n_clusters)
plt.show()

mnistcluster.assess(y)



# predict
x_test, y_test = data.load_mnist_test()

preds_2 = mnistcluster.predict(x_test)
mnistcluster.assess(y_test)


manifoldGMM2 = n2d.UmapGMM(n_clusters, umap_neighbors=10)
mnistcluster2 = n2d.n2d(x.shape[-1], manifoldGMM2, n_clusters)

mnistcluster2.fit(x, pretrain_epochs=10)

n2d.save_n2d(mnistcluster2, 'test_ae.h5', 'test_man.sav')
