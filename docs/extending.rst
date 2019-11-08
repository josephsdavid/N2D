Advanced Usage
========================

As mentioned earlier, N2D is an entirely extensible framework for not too deep clustering. In this section we will discuss modifying the clustering/manifold learning methods, and modifying the autoencoder. The independence of each step of N2D means we can change the autoencoder into a convolutional autoencoder or some other more complex LSTM based autoencoder, depending on the application, or change clustering method. We will discuss changing both parts of the algorithm below.

Changing the Manifold Clustering Step
------------------------------------------

To change the clustering or manifold learning method, we do not need to do much. We simply need to write a class which contains a `predict` method, which takes in the autoencoded embedding. In this example, let us assume we want to do spectral clustering, again on the HAR dataset. Lets first go ahead and set up the autoencoder using weights made in the previous section, and load up our data. ::
        
        
      import n2d
      import matplotlib
      import matplotlib.pyplot as plt
      import seaborn as sns
      plt.style.use(['seaborn-white', 'seaborn-paper'])
      sns.set_context("paper", font_scale = 1.3)
      matplotlib.use('agg')
      import umap
      from sklearn.cluster import SpectralClustering
      from n2d import datasets as data
      # load in the data
      x, y, y_names = data.load_har()
      n_clusters = 6

      # set up deep part
      harcluster = n2d.n2d(x, nclust = n_clusters)
      harcluster.preTrainEncoder(weights = "weights/har-1000-ae_weights.h5")


Now, we need to write a class for clustering the manifold. Presented below is a general framework for doing so. ::

        class UmapSpectral:

                def __init__(self, nclust,
                             umapdim = 2,
                             umapN = 10,
                             umapMd = float(0),
                             umapMetric = 'euclidean',
                             random_state = 0):
                    # store the number of clusters for easy reference
                    self.nclust = nclust

                    # change this bit for changing the manifold learner
                    # this part initializes UMAP
                    self.manifoldInEmbedding = umap.UMAP(
                        random_state = random_state,
                        metric = umapMetric,
                        n_components = umapdim,
                        n_neighbors = umapN,
                        min_dist = umapMd
                    )

                    # change this bit to change the clustering mechanism
                    # this initializes the clustering algorithm in general
                    self.clusterManifold = SpectralClustering(
                		n_clusters = nclust,
                		affinity = 'nearest_neighbors',
                		random_state = random_state
                	)

                    # You need this always, this sets up a space to store
                    # the manifold of the autoencoded embedding
                    self.hle = None


                def predict(self, hl):
                # if you change the learner, you need to change the predict
                # method
                    self.hle = self.manifoldInEmbedding.fit_transform(hl)
                    self.clusterManifold.fit(self.hle)
                    y_pred = self.clusterManifold.fit_predict(self.hle)
                    return(y_pred)


And there you go! We have made a class that N2D can take in once initialized, and are ready for action. ::
        
        manifoldSC = UmapSpectral(6)
        harcluster.predict(manifoldSC)



Replacing The Autoencoder
-------------------------------


This is a slightly trickier part, but it is still not too hard. We again need to write a class, this time for our new autoencoder. The class needs to have two special properties: **The first argument needs to be the dimensions of the autoencoder**, and it needs to have a pretrain method. For now, this is still limited to Keras/Tensorflow models, but in the future we will allow for full exchange of the backend to for example pytorch. Here we will show an example of a denoising autoencoder. ::

        
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
        from keras import backend as K
        import sys
        from keras.layers import Dense, Input
        from keras.models import Model



        class denoisingAutoEncoder:
        # all this stuff is pretty standard
            def __init__(self, dims, noise_factor = 0.5, act = 'relu'):
                self.noise_factor = noise_factor
                self.dims = dims
                self.act = act
                # input layer
                self.x = Input(shape = (dims[0],), name = 'input')
                self.h = self.x
                n_stacks = len(self.dims) - 1

                # make the nice symmetric network 
                # obviously this bit can be changed, but here
                # we construct the network in a nice way
                for i in range(n_stacks - 1):
                    self.h = Dense(self.dims[i + 1], activation = self.act, name = 'encoder_%d' %i)(self.h)
                self.h = Dense(self.dims[-1], name = 'encoder_%d' % (n_stacks -1))(self.h)
                for i in range(n_stacks - 1, 0, -1):
                    self.h = Dense(self.dims[i], activation = self.act, name = 'decoder_%d' % i )(self.h)
                self.h = Dense(dims[0], name = 'decoder_0')(self.h)
        
                self.Model = Model(inputs = self.x, outputs = self.h)

            def add_noise(self, x):
            	# this is the new bit
                x_noisy = x + self.noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x.shape)
                x_noisy = np.clip(x_noisy, 0., 1.)

                return x_noisy


                def pretrain(self, dataset, batch_size = 256, pretrain_epochs = 1000,
                                 loss = 'mse', optimizer = 'adam',weights = None,
                                 verbose = 0, weightname = 'fashion'):
                    if weights == None:
                        self.Model.compile(
                            loss = loss, optimizer = optimizer
                        )
                        self.Model.fit(
                            self.add_noise(x), x,
                            batch_size = batch_size,
                            epochs = pretrain_epochs
                        )

                        self.Model.save_weights("weights/" + weightname + "-" +
                                                str(pretrain_epochs) +
                                                "-ae_weights.h5")
                    else:
                        self.Model.load_weights(weights)


And there we go! We can now run our denoising autoencoder just as we would normally. Note that the design of this code is pretty standard and integrates well with the framework, and should be followed pretty closely for now. On later versions this will be made easier to extend. Lets go ahead and load up some data! :: 
        
        x,y, y_names = data.load_har()
        n_clusters = 6
        model = n2d.n2d(x, autoencoder = denoisingAutoEncoder, nclust = n_clusters, ae_args={'noise_factor': 0.5})

A brief note on how to extend the autoencoder. The input to the autoencoder argument is a **Class**, and any extra arguments to be passed into the autoencoder go in a **dict** in ae_args. Now, lets go ahead and pretrain our model just as normal. ::
        
        
        model.preTrainEncoder(weight_id="fashion_denoise")
        
        
        manifoldGMM = n2d.UmapGMM(n_clusters)
        
        model.predict(manifoldGMM)
        
        model.visualize(y, names=None, dataset = "fashion_denoise", nclust = n_clusters)
        print(model.assess(y))


 
