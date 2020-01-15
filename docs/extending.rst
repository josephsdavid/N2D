Advanced Usage
========================

As mentioned earlier, N2D is an entirely extensible framework for not too deep clustering. In this section we will discuss modifying the clustering/manifold learning methods, and modifying the autoencoder. The independence of each step of N2D means we can change the autoencoder into a convolutional autoencoder or some other more complex LSTM based autoencoder, depending on the application, or change clustering method. We will discuss changing both parts of the algorithm below.

Changing the Manifold Clustering Step
------------------------------------------

To change the manifold learner or clustering algorithm, we must write a new class. This class **Needs** a fit and a predict method, otherwise everything else is up to you!

Lets assume for our use case, k means will work very well. Now lets write a k means clustering class!::

        
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

And there you go! We have made a class that N2D can take in once initialized, and are ready for action. :: 
        
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


Did not do half bad! It performed almost immesurably worse than the original clustering mechanism

Replacing The Autoencoder
-------------------------------

This is slightly more involved, but still pretty easy! The autoencoder needs to have at least three arguments: **data**, **ndim**, and **architecture**. It will also need a fit method, a **Model** attribute which represents the entire network, and an **encoder** attribute which represents the encoder part of the autoencoder. This allows us to build the autoencoder in a programatic fashion. It also needs a fit method. Below, for a simple example, we will build a denoising autoencoder ::


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
        import umap
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import model
        
        x,y, y_names = data.load_fashion()
        
        
        
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

                if weights is None: # if there are no weights to load for the encoder, make encoder
                    self.Model.compile(
                        loss=loss, optimizer=optimizer
                    )

                    if weight_id is not None: # if we are going to save the weights somewhere
                        if patience is not None: #if we are going to do early stopping
                            callbacks = [EarlyStopping(monitor='loss', patience=patience),
                                         ModelCheckpoint(filepath=weight_id,
                                                         monitor='loss',
                                                         save_best_only=True)]
                        else:
                            callbacks = [ModelCheckpoint(filepath=weight_id,
                                                         monitor='loss',
                                                         save_best_only=True)]
                        # fit the model with the callbacks
                        self.Model.fit(
                            x_noisy, x,
                            batch_size=batch_size,
                            epochs=pretrain_epochs,
                            callbacks=callbacks, verbose=verbose
                        )
                        self.Model.save_weights(weight_id)
                    else: # if we are not saving weights
                        if patience is not None:
                            callbacks = [EarlyStopping(monitor='loss', patience=patience)]
                            self.Model.fit(
                                x_noisy, x,
                                batch_size=batch_size,
                                epochs=pretrain_epochs,
                                callbacks=callbacks, verbose=verbose
                            )
                        else:
                            self.Model.fit(
                                x_noisy, x,
                                batch_size=batch_size,
                                epochs=pretrain_epochs,
                                verbose=verbose
                            )
                else: # otherwise load weights
                    self.Model.load_weights(weights)
Again, this code is big, but basically the new class you define needs to build the autoencoder in the __init__ method, it needs to save the encoder network as self.encoder, and it needs to have a predict method. Extra arguments can be put at the end, as they will go into the *ae_args* dict



Lets go ahead and show how we can use the new autoencoder! Please refer to the table in the previous chapter for all the arguments for the N2D class. ::


        x,y, y_names = data.load_fashion()
        
        n_clusters = 10
        
        model = n2d.n2d(x.shape[-1], manifold_learner=n2d.UmapGMM(n_clusters),
                        autoencoder = denoisingAutoEncoder, ae_dim = n_clusters,
                        ae_args={'noise_factor': 0.5})
        
        model.fit(x, weights="weights/fashion_denoise-1000-ae_weights.h5")
        
        denoising_preds = model.predict(x)
        


It is important to note that when you initialize the N2D class, it takes in an **already initialized manifold clusterer**, and just the **class** of the autoencoder. This  is because the manifold clustering may have many varying arguments, as it contains two steps which will change in arguments, while an autoencoder can be constructed just by specifying the dimensions. The extra arguments to the autoencoder, if you need them, are passed in through the ae_args dict.
