Advanced Usage
========================

As mentioned earlier, N2D is an entirely extensible framework for not too deep clustering. In this section we will discuss modifying the clustering/manifold learning methods, and modifying the autoencoder. The independence of each step of N2D means we can change the autoencoder into a convolutional autoencoder or some other more complex LSTM based autoencoder, depending on the application, or change clustering method. We will discuss changing both parts of the algorithm below.

Changing the Manifold Clustering Step
------------------------------------------

To change the manifold learner or clustering algorithm, we must write a new class. This class **Needs** a fit and a predict method, otherwise everything else is up to you!

Lets assume Umap and Spectral Clustering is an idea we are interested in (its not a great one). Let's write a class for clustering the manifold. You can use this as a general framework for your own use ::


        from sklearn.cluster import SpectralClustering
        import umap
        class UmapSpectral:
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
        	self.clusterManifold = SpectralClustering(
        		n_clusters = nclust,
        		affinity = 'nearest_neighbors',
        		random_state = random_state
        	)
        
        	self.hle = None
        
        
            def fit(self, hl):
                self.hle = self.manifoldInEmbedding.fit_transform(hl)
                self.clusterManifold.fit(self.hle)
        
            def predict(self):
            # obviously if you change the clustering method or the manifold learner
            # youll want to change the predict method too.
        	y_pred = self.clusterManifold.fit_predict(self.hle)
                return(y_pred)

And there you go! We have made a class that N2D can take in once initialized, and are ready for action. ::
        
        import n2d
        from n2d import datasets as data
        x, y, y_names = data.load_har()

        manifoldSC = UmapSpectral(6)
        SCclust = n2d.n2d(x, manifoldSC, ndim = 6)

        # now we can continue as normal!

        SCclust.fit(weights = "weights/har-1000-ae_weights.h5")
        SCclust.predict()
        print(SCclust.assess(y))
        # (0.40946, 0.42137, 0.14973)




Replacing The Autoencoder
-------------------------------

This is slightly more involved, but still pretty easy! The autoencoder needs to have at least three arguments: **data**, **ndim**, and **architecture**. This allows us to build the autoencoder in a programatic fashion. It also needs a fit method. Below, for a simple example, we will build a denoising autoencoder ::


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
        from keras import backend as k
        
        import tensorflow as tf
        import sys
        import umap
        from keras.layers import dense, input
        from keras.models import model
        
        x,y, y_names = data.load_fashion()
        
        
        class denoisingAutoEncoder:
            def __init__(self, data, ndim, architecture,
            noise_factor = 0.5, act = 'relu'):
                dims = [data.shape[-1]] + architecture + [ndim]
                self.dims = dims
                self.noise_factor = noise_factor
                self.act = act
                self.x = Input(shape = (dims[0],), name = 'input')
                self.h = self.x
                n_stacks = len(self.dims) - 1
                for i in range(n_stacks - 1):
                    self.h = Dense(self.dims[i + 1], activation = self.act, name = 'encoder_%d' %i)(self.h)
                self.h = Dense(self.dims[-1], name = 'encoder_%d' % (n_stacks -1))(self.h)
                for i in range(n_stacks - 1, 0, -1):
                    self.h = Dense(self.dims[i], activation = self.act, name = 'decoder_%d' % i )(self.h)
                self.h = Dense(dims[0], name = 'decoder_0')(self.h)
        
                self.Model = Model(inputs = self.x, outputs = self.h)
        
            def add_noise(self, x):
            	# this is the new bit
                x_clean = x
                x_noisy = x_clean + self.noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_clean.shape)
                x_noisy = np.clip(x_noisy, 0., 1.)
        
                return x_clean, x_noisy
        
            def fit(self, dataset, batch_size = 256, pretrain_epochs = 1000,
                             loss = 'mse', optimizer = 'adam',weights = None,
                             verbose = 0, weightname = 'fashion'):
                if weights == None:
                    x, x_noisy = self.add_noise(dataset)
                    self.Model.compile(
                        loss = loss, optimizer = optimizer
                    )
                    self.Model.fit(
                        x_noisy, x,
                        batch_size = batch_size,
                        epochs = pretrain_epochs
                    )
        
                    self.Model.save_weights("weights/" + weightname + "-" +
                                            str(pretrain_epochs) +
                                            "-ae_weights.h5")
                else:
                    self.Model.load_weights(weights)


Again, this code is big, but basically the new class you define needs to build the autoencoder in the __init__ method, and it needs to have a method which fits the autoencoder. The rest is again up to you!



Lets go ahead and show how we can use the new autoencoder! Please refer to the table in the previous chapter for all the arguments for the N2D class. ::


        x,y, y_names = data.load_fashion()
        
        n_clusters = 10
        
        model = n2d.n2d(x, manifoldLearner=n2d.UmapGMM(n_clusters),
        	autoencoder = denoisingAutoEncoder, 
        	ndim = n_clusters, ae_args={'noise_factor': 0.5, 'act':'relu'})
        
        model.fit(weight_id="fashion_denoise")
        
        model.predict()
        
        model.visualize(y, y_names, savePath = "viz/fashion_denoise", nclust = n_clusters)
        print(model.assess(y))
        


It is important to note that when you initialize the N2D class, it takes in an **already initialized manifold clusterer**, and just the **class** of the autoencoder. This  is because the manifold clustering may have many varying arguments, as it contains two steps which will change in arguments, while an autoencoder can be constructed just by specifying the dimensions. The extra arguments to the autoencoder, if you need them, are passed in through the ae_args dict.
