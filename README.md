[![Documentation Status](https://readthedocs.org/projects/n2d/badge/?version=latest)](https://n2d.readthedocs.io/en/latest/?badge=latest)

Welcome to the stable master branch of N2D!! The active development branch is on [dev](https://github.com/josephsdavid/N2D/tree/dev), if you are contributing, please make branches from that!

# Changes

* [Model saving and loading!](https://n2d.readthedocs.io/en/latest/quickstart.html#saving-and-loading)
- [*] TF2 ready


# Not Too Deep Clustering

This is a library implementation of [n2d](https://github.com/rymc/n2d). To learn more about N2D, and clustering manifolds of autoencoded embeddings, please refer to the [amazing paper](https://arxiv.org/abs/1908.05968) published August 2019.

## What is it?

Not too deep clustering is a state of the art "deep" clustering technique, in which first, the data is embedded using an autoencoder. Then, instead of clustering that using some deep clustering network, we use a manifold learner to find the underlying (local) manifold in the embedding. Then, we cluster that manifold. In the paper, this was shown to produce high quality clusters without the standard extreme feature engineering required for clustering.

In this repository, a framework for A) reproducing the study and B) extending the study is given, for further research and use in a variety of applications

# Documentation

Full documentation is available on [read the docs](https://n2d.readthedocs.io/en/latest/)

# Installation

N2D is [available on pypi](https://pypi.org/project/n2d/)

```sh
pip install n2d
```

# Usage

First, lets load in some data. In this example, we will use the Human Activity Recognition(HAR) dataset. In this dataset, sets of time series with data from mobile devices is used to classify what the person is doing (walking, sitting, etc.)

```python
from n2d import datasets as data
x,y, y_names = data.load_har()
```

Next, lets set up our deep learning environment, as well as load in necessary libraries:

```python
import os
import random as rn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
matplotlib.use('agg')

import tensorflow as tf

# set up environment
os.environ['PYTHONHASHSEED'] = '0'


rn.seed(0)

np.random.seed(0)
```

Finally, we are ready to get clustering!

First, we need to define the manifold learning and clustering algorithm which we will use to cluster the autoencoded embedding. In general, it is best to use UmapGMM, which in the paper gave the absolute best performance.

```python
import n2d as nd

n_clusters = 6  #there are 6 classes in HAR

manifoldGMM = n2d.UmapGMM(n_clusters) 
```

The next step in this framework is to initialize the n2d object, which builds an autoencoder network and gets everything ready for clustering:

```python
harcluster = n2d.n2d(x.shape[-1], manifoldGMM, ae_dim = n_clusters)
```

Next, we fit the data. In this step, the autoencoder is trained on the data, setting up weights.

```python
harcluster.fit(x, weight_id = "har")
```

The next time we want to use this autoencoder, we will instead use the weights argument:

```python
harcluster.fit(x, weights = "har-1000-ae_weights.h5")
```

Now we can make a prediction, as well as visualize and assess. In this step, the manifold learner learns the manifold for the data, which is then clustered. By default, it makes the prediction on the data stored internally, however you can specify a new `x` in order to make predictions on new data.

```python
preds = harcluster.predict(x)
# predictions are stored in harcluster.preds
harcluster.visualize(y, y_names, n_clusters = n_clusters)
print(harcluster.assess(y))
# (0.81212, 0.71669, 0.64013)
```

Before viewing the results, lets talk about the metrics. The first metric is cluster accuracy, which we see here is 81.2%, which is absolutely state of the art for the HAR dataset. The next metric is NMI, which is another metric which describes cluster quality based on labels, independent of the number of clusters. We have an NMI of 0.717, which is again absolutely state of the art for this dataset. The last metric, ARI, shows another comparison between the actual groupings and our grouping. A value of 1 means the groupings are nearly the same, while a value of 0 means they completely disagree. We have a value of 0.64013, which indicates that are predictions are more or less in agreement with the truth, however they are not perfect.

![N2D prediction](https://i.imgur.com/91iwVVj.png)

N2D prediction

![](https://i.imgur.com/8PTPTmE.png)
Actual clusters

## Extending

### Replacing the manifold clustering mechanism

So far, this framework only includes the method for manifold clustering which the authors of the paper deemed best, umap with gaussian mixture clustering. Lets say however we want to try out K means instead

```python
import n2d
import numpy as np
from sklearn.cluster import KMeans
import umap

from n2d import datasets as data

x, y, y_names = data.load_har()

class UmapKmeans:
    # you can pass whatever parameters you need to here
    def __init__(self, n_clusters,
                 umap_dim=2,
                 umap_neighbors=10,
                 umap_min_distance=float(0),
                 umap_metric='euclidean',
                 random_state=0
                 ):
        # This parameter is not necessary but i find it useful 
        self.n_clusters = n_clusters
        
        # this is how I generally structure this code, easy to modify
        self.manifold_in_embedding = umap.UMAP(
            random_state=random_state,
            metric=umap_metric,
            n_components=umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_distance
        )

        self.cluster_manifold = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_jobs=-1
        )
        self.hle = None
    
    # fit method takes in one argument, the embedding! important
    def fit(self, hl):
        self.hle = self.manifold_in_embedding.fit_transform(hl)
        self.cluster_manifold.fit(self.hle)
    
    # takes in the embedding!
    def predict(self, hl):
        manifold = self.manifold_in_embedding.transform(hl)
        y_pred = self.cluster_manifold.predict(manifold)
        return(np.asarray(y_pred))

    # takes in the embedding!
    def fit_predict(self, hl):
        self.hle = self.manifold_in_embedding.fit_transform(hl)
        self.cluster_manifold.fit(self.hle)
        y_pred = self.cluster_manifold.predict(self.hle)
        return(np.asarray(y_pred))
```

Now we can run and assess our new clustering method:

```python
import n2d
from n2d import datasets as data
import matplotlib.pyplot as plt
x, y, y_names = data.load_har()

manifoldKM = UmapKmeans(6)
kmclust = n2d.n2d(x.shape[-1], manifoldKM, 6)

# now we can continue as normal!

kmclust.fit(x, weights="weights/har-1000-ae_weights.h5")

_ = kmclust.predict(x)
print(kmclust.assess(y))
# (0.81668, 0.71208, 0.64484) 
```

This clearly did not go as well, however we can see that it is very easy to extend this library. We could also try out swapping UMAP for ISOMAP, the clustering method with kmeans, or maybe with a deep clustering technique. 

### Replacing the embedding mechanism

We can also replace the embedding learner, by writing a new class. In this example we will implement a denoising autoencoder, as demonstrated in [this awesome blog post](https://blog.keras.io/building-autoencoders-in-keras.html)

```python
import os
import n2d
from n2d import datasets as data
import random as rn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
matplotlib.use('agg')
import tensorflow as tf
import sys
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


x,y, y_names = data.load_fashion()

# we decide to use this for some strange reason
class denoisingAutoEncoder:
    def __init__(self, input_dim, output_dim, architecture, noise_factor = 0.5, act='relu'):
        self.noise_factor = noise_factor
        shape = [input_dim] + architecture + [output_dim]
        self.dims = shape
        self.act = act
        self.x = Input(shape=(self.dims[0],), name='input')
        self.h = self.x
        n_stacks = len(self.dims) - 1

        # this is how I like to set up the networkm however however you want to do it it doesnt matter.
        # it NEEDS to have a self.encoder attribute
        for i in range(n_stacks - 1):
            self.h = Dense(
                self.dims[i + 1], activation=self.act, name='encoder_%d' % i)(self.h)
        self.encoder = Dense(
            self.dims[-1], name='encoder_%d' % (n_stacks - 1))(self.h)
        self.decoded = Dense(
            self.dims[-2], name='decoder', activation=self.act)(self.encoder)
        for i in range(n_stacks - 2, 0, -1):
            self.decoded = Dense(
                self.dims[i], activation=self.act, name='decoder_%d' % i)(self.decoded)
        self.decoded = Dense(self.dims[0], name='decoder_0')(self.decoded)

        self.Model = Model(inputs=self.x, outputs=self.decoded)

        # NEEDED!!
        self.encoder = Model(inputs=self.x, outputs=self.encoder)


    def add_noise(self, x):
        x_clean = x
        x_noisy = x_clean + self.noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_clean.shape)
        x_noisy = np.clip(x_noisy, 0., 1.)

        return x_clean, x_noisy

    def fit(self, x, batch_size = 256, pretrain_epochs = 1000,
                     loss = 'mse', optimizer = 'adam',weights = None,
                     verbose = 0, weight_id = 'fashion', patience = None):

        x, x_noisy = self.add_noise(x)
        if weights is None:
            self.Model.compile(
                loss=loss, optimizer=optimizer
            )
            if patience is not None:
                callbacks = [EarlyStopping(monitor='loss', patience=patience),
                             ModelCheckpoint(filepath=weight_id,
                                             monitor='loss',
                                             save_best_only=True)]
            else:
                callbacks = [ModelCheckpoint(filepath=weight_id,
                                             monitor='loss',
                                             save_best_only=True)]
            self.Model.fit(
                x_noisy, x,
                batch_size=batch_size,
                epochs=pretrain_epochs,
                callbacks=callbacks, verbose=verbose
            )

            self.Model.save_weights(weight_id)
        else:
            self.Model.load_weights(weights)

 x,y, y_names = data.load_fashion()
 
 n_clusters = 10
 
 model = n2d.n2d(x.shape[-1], manifold_learner=n2d.UmapGMM(n_clusters),
                 autoencoder = denoisingAutoEncoder, ae_dim = n_clusters,
                 ae_args={'noise_factor': 0.5})
 
 model.fit(x, weights="weights/fashion_denoise-1000-ae_weights.h5")
 
 denoising_preds = model.predict(x)

```

Lets talk about the ingredients we need for modification:

The class needs to take in a list of dimensions for the model. These dimensions should only include the center layers (read, not input and output). If we want to change the model architecture, we simply put that as the `architecture` argunent of n2d:

```python
n2d.n2d(..., architecture = [500,500,2000])
```

This will design the networks to be [input dimensions, 500, 500, 2000, output dimensions], as seen in the denoisingAutoencoder class. If we want to change the autoencoder itself, we need to write a class which accepts the shape, and then some other (preferably with defaults) arguments. This class NEEDS to have a method called fit. Everything else is up to you! To add in extra arguments to whatever your new autoencoder is, you pass them in through a dict called ae_args, as seen in the above example.


# Roadmap

- [x] Package library
- [x] Package data
- [x] Early stopping
- [ ] Implement data augmentation techniques for images, sequences, and time series
- [x] Make autoencoder interchangeable just like the rest
- [x] Simpler way to extract embedding
- [ ] Implement other types of autoencoders as well as convolutional layers
- [x] Manage file saving paths better
- [ ] Implement other promising methods
- [ ] Make assessment/visualization more extensible
- [x] Documentation?
- [x] Find an elegant way to deal with pre training weights
- [ ] Package on Nix
- [ ] Blog post?
- [ ] Clean up code examples!

# Contributing

N2D is a work in progress and is open to suggestions to make it faster, more extensible, and generally more usable. Please make an issue if you have any ideas of how to be more accessible and more usable! 


# Citation

If you use N2D in your research, please credit the original authors of the paper. Bibtex included below:

```
@article{2019arXiv190805968M,
  title = {N2D:(Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding},
  author = {{McConville}, Ryan and {Santos-Rodriguez}, Raul and {Piechocki}, Robert J and {Craddock}, Ian},
  journal = {arXiv preprint arXiv:1908.05968},
  year = "2019",
}
```
