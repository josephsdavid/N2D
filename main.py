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

# set up stuff




os.environ['PYTHONHASHSEED'] = '0'


rn.seed(0)
tf.set_random_seed(0)
np.random.seed(0)

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    print("Using GPU")
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1,
                                  )
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)



x,y, y_names = nd.load_har()

n_clusters = 6
harcluster = nd.n2d(x, nclust = n_clusters)

harcluster.preTrainEncoder(weights = "har-1000-ae_weights.h5")

manifold = nd.UmapGMM(n_clusters)

harcluster.predict(manifold)

harcluster.visualize(y, y_names, dataset = "har", nclust = n_clusters)
print(harcluster.assess(y))
