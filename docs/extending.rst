Advanced Usage
========================

As mentioned earlier, N2D is an entirely extensible framework for not too deep clustering. In this section we will discuss modifying the clustering/manifold learning methods, and modifying the autoencoder. The independence of each step of N2D means we can change the autoencoder into a convolutional autoencoder or some other more complex LSTM based autoencoder, depending on the application, or change clustering method. We will discuss changing both parts of the algorithm below.

Changing the Manifold Clustering Step: 
------------------------------------------

To extend N2D to include your favorite autoencoder or clustering algorithm, you can use either of the two **generator** classes. To replace the manifold clustering step, we use the **manifold_cluster_generator** class. This class takes in 4 arguments:

#. The class of the manifold learner, for example, umap.UMAP
#. A dict of arguments to initialize the manifold learner with
#. The class of the clusterer
#. A dict of arguments for the clusterer

Objects created by **generators** can be passed directly into N2D, without needing any boilerplate code. Lets go ahead and look at an example. Let us assume that we want to use density based clustering with UMAP and our standard autoencoder based dimensionality reduction. First, we import our libraries:::


        import n2d
        import numpy as np
        import n2d.datasets as data
        import hdbscan
        import umap

        x, y = data.load_mnist()

 First, we make our autoencoder, for now using the AutoEncoder class:::
        
        ae = n2d.AutoEncoder(input_dim = x.shape[-1], latent_dim = 20) # chosen arbitrarily

Next, lets define the arguments we wish to initialize hdbscan and umap with. Please note these values are chosen either arbitrarily or for visualization:::

        # hdbscan arguments
        hdbscan_args = {"min_samples":10,"min_cluster_size":500, 'prediction_data':True}

        # umap arguments
        umap_args = {"metric":"euclidean", "n_components":2, "n_neighbors":30,"min_dist":0}

Next, lets go ahead and generate something we can use to cluster our embedding!!::

        db_clust = n2d.manifold_cluster_generator(umap.UMAP, umap_args, hdbscan.HDBSCAN, hdbscan_args)

Now we pass those into **n2d.n2d** and we are good to go!::

        n2d_db = n2d.n2d(ae, db_clust)

We can fit as usual::

        n2d_db.fit(x, epochs = 10) # for times sake, this is just an example


Because this is dbscan, after fitting we can say we are done! The fitted n2d object can do anything the parent clustering class can do (it also shares its limitations). This means that we can just go ahead and grab the predictions which hdbscan already so kindly made for us:::

        # the probabilities 
        print(n2d_db.clusterer.probabilities_)
        # the labels
        print(n2d_db.clusterer.labels_)

The clustering algorithm is stored in **.clusterer**, while the manifold learner is stored in **.manifolder**::
        print(n2d_db.clusterer)
        print(n2d_db.manifolder)

Note that while our fitted n2d object has all the attributes of the clustering mechanism, it also has all of the limitations. That means, in the case of hdbscan, we can do **fit_predict**, however there is no **predict** method.::

        # works
        n2d_db.fit_predict(x, epochs = 10)
        # fails
        n2d_db.predict(x)

However, hdbscan has a neat trick where we can make "approximate predictions". This is allowed! We can write a imple function to get the approximate predictions and make predictions on new data:::

        x_test, y_test = data.load_mnist_test()

        # predict on new data with dbscan and not too deep clustering!
        def approx_predict(n2d_obj, newdata):
                embedding = n2d_obj.encoder.predict(newdata)
                manifold = n2d_obj.manifolder.transorm(embedding)
                labs, probs = hdbscan.approximate_predict(n2d_obj.clusterer, manifold)
                return labs, probs

        labs, probs = approx_predict(n2d_db, x_test)

 Next, lets look at swapping out the autoencoder!!


Changing the Autoencoder
----------------------------------------------

To swap out the autoencoder, we can, just as with the clustering step, use a **generator** class. In this case, we will use the **autoencoder_generator** class. This class takes in 2 things: an iterable of model parts, and if needed a lambda function. The lambda function is not necessary, and by default does nothing. However, for some use cases it may be useful to change the inputs to the encoder. We will look at one such case: a denoising autoencoder. **NOTE: this is a simple example to showcase features, there is no real precedent for clustering with a denoising autoencoder**

First, again, we load up our libraries:::

        import n2d
        from n2d import datasets as data
        from tensorflow.keras.layers import Dense, Input
        import seaborn as sns
        import umap
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        sns.set_context("paper", font_scale=1.3)

        x, y, y_names = data.load_fashion()

        n_clusters = 10

Next, as usual, we are going to make our autoencoder, however this time without the AutoEncoder class. We are going to want to make a list, tuple, or array that contains pointers to the input layer, the end of the encoder (center layer), and output layer of the encoder. To do that we will use the tf.keras functional API:::

        hidden_dims = [500, 500, 2000]
        input_dim = x.shape[-1]
        inputs = Input(input_dim)
        encoded = inputs
        for d in hidden_dims:
            encoded = Dense(d, activation = "relu")(encoded)
        encoded = Dense(n_clusters)(encoded)
        decoded = encoded
        for d in hidden_dims[::-1]:
            decoded = Dense(d, activation = "relu")(decoded)
        outputs = Dense(input_dim)(decoded)

Lets go ahead and define our first set of inputs for the **autoencoder_generator** class:::
        
        ae_stages = (inputs, encoded, outputs)

Again, the autoencoder_generator class requires an iterable containing the input layer, the encoding, and the decoded output layer of the model. The rest is taken care of internally. As this is a denoising autoencoder, lets also write a function that adds noise to our data:::

        def add_noise(x, noise_factor):
            x_clean = x
            x_noisy = x_clean + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_clean.shape)
            x_noisy = np.clip(x_noisy, 0., 1.)
            return x_noisy


Now we can go ahead and generate an autoencoder for N2D:::
      
        denoising_ae = n2d.autoencoder_generator(ae_stages, x_lambda = lambda x: add_noise(x, 0.5))

Finally, lets initialize UmapGMM and our model, and make a quick prediction:::

        umapgmm = n2d.UmapGMM(n_clusters)
        model = n2d.n2d(denousing_ae, umapgmm)
        model.fit(x, epochs=10)
        model.predict(x)
        model.visualize(y, y_names,  n_clusters = n_clusters)
        plt.show()
        print(model.assess(y))


And with that, you are ready to get clustering and testing new and unexplored algorithms! If you are having any troubles, or ideas for features, please make an issue on github!!
