import n2d
from n2d import datasets as data
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
import random as rn
rn.seed(0)
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
np.random.seed(0)


x,y = data.load_mnist()

n_clusters = 10
manifoldGMM = n2d.UmapGMM(n_clusters, umapN=10)
mnistcluster = n2d.n2d(input_dim = x.shape[-1], manifoldLearner =  manifoldGMM, ae_dim = n_clusters)

# fit
mnistcluster.fit(x,weights = "weights/mnist-1000-ae_weights.h5", patience = None)
preds_0 = mniscluster.predict(x)

# fit_predict
preds_1 = mnistcluster.fit_predict(x,weights = "weights/mnist-1000-ae_weights.h5", patience = None)
mnistcluster.visualize(y, None, nclust = n_clusters)
plt.show()

mnistcluster.assess(y)



# predict
x_test, y_test = data.load_mnist_test()

preds_2 = mnistcluster.predict(x_test)
mnistcluster.assess(y_test)
