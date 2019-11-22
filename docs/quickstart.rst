Getting started
========================

Here we will talk about getting started with N2D so you can get clustering!!

Installation
--------------

N2D is on Pypi and readily installable ::

        pip install n2d



Loading Data
----------------

N2D comes with **5** built in datasets: 3 image datasets and two time series datasets, described below:

* MNIST
  - **Description**: Standard handwritten image dataset. 10 classes
* MNIST-Test
  - **Description**: Test set of MNIST. 10 classes
* MNIST-Fashion
  - **Description**: Pictures of articles of clothing, similar to MNIST but much more difficult. 10 classes
* Human Activity Recognition (HAR)
  - **Description**: Time series of accelerometer data, used to determine whether the recorded human is sitting, walking, going upstairs/downstairs etc. 6 classes
* Pendigits
  - **Description**: Pressure sensor data of humans writing. Used to determine what number the human is writing. 10 classes

To actually load the data, we import the datasets from n2d, shown below along with the data import functions and their outputs ::

       from n2d import datasets as data

       # imports mnist
       data.load_mnist() # x, y 

       # imports mnist_test
       data.load_mnist_test() # x, y

       # imports fashion
       data.load_fashion() # x, y, y_names

       # imports HAR
       data.load_har() # x, y, y_names

       # imports pendigits
       data.load_pendigits # x, y



In this example, we are going to use HAR. ::

        x, y, y_names = data.load_har()


Building the model
---------------------


Next, we want to setup the model. In this example, we are going to use the defaults, so we can quickly get clustering! Later on, we will discuss extending and changing the inner workings of the algorithm (details such as architecture, autoencoder type, clustering mechanism, etc). Lets first import some libraries and again load in our data ::
        
      import n2d
      import matplotlib
      import matplotlib.pyplot as plt
      import seaborn as sns
      plt.style.use(['seaborn-white', 'seaborn-paper'])
      sns.set_context("paper", font_scale = 1.3)
      matplotlib.use('agg')

      # for reproducibiliry
      import os
      os.environ['PYTHONHASHSEED'] = '0'
      os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
      import random as rn
      rn.seed(0)
      import tensorflow as tf
      tf.set_random_seed(0)
      import numpy as np
      np.random.seed(0)

      from n2d import datasets as data
      # load in the data
      x, y, y_names = data.load_har()


Next, we need to initialize the N2D object. This requires three arguments: the number of dimensions we would like to represent the data as, a manifold clustering mechanism, and the data (without labels). ::
        
        n_clusters = 6
        manifoldGMM = n2d.UmapGMM(n_clusters)
        harcluster = n2d.n2d(x, manifoldGMM, ndim = n_clusters)



First, lets talk about **n2d.UmapGMM**. This is the main clustering and manifold learning tool in the whole library, and should be understood well.


Clustering the Embedded Manifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lets talk a bit more about how we learn the manifold and cluster it!! This is done primarily with the UmapGMM object ::
        
        manifoldGMM = n2d.UmapGMM(n_clusters)

This initializes the hybrid manifold learner/clustering arguments. In general, UmapGMM performs best, but in a later section we will talk about replacing it with other clustering/manifold learning techniques. The arguments for UmapGMM are shown below:


.. list-table:: UmapGMM Arguments
        :widths: 25 25 25
        :header-rows: 1

        * - Argument
          - Default
          - Description
        * - nclust
          - no default
          - The number of clusters
        * - umapdim
          - 2
          - Number of dimensions of the manifold.
        * - umapN
          - 10
          - Number of nearest neighbors to consider for UMAP. Defaults to 10, to recreate cutting edge results shown in the paper, however often 20 is a better value 
        * - umapMd
          - float(0)
          - Minimum distance between points within the manifold. Smaller numbers get tighter, better clusters while larger numbers are better for visualization
        * - umapMetric
          - 'euclidean'
          - The distance metric to use for UMAP.
        * - random_state
          - 0
          - The random seed

For our use case, there are two main tunables: **umapdim**, and **umapN**. **umapdim** is the number of dimensions you wish to project the autoencoded embedding in. In general, values between **2** and **the number of clusters** are acceptable. It is best to start at 2 (the default value) and then go up from there. All of the breakthrough results in the paper were done with umapdim =2.  **umapN** is the number of nearest neighbors UMAP will use when constructing its KNN graph. In the case of N2D, this should be a small value, as we want to learn the **local manifold**. The default value for umapN is **10**, as it will allow you to reproduce the results in the paper, however umapN = **20** sometimes performs slightly better, *especially if the autoencoder loss is high*. Since umapGMM takes just a few seconds to run, it is worth it to tune these two values in general.

Initializing N2D
~~~~~~~~~~~~~~~~~~~~~~~~~
Next, we initialize the **n2d** object. Upon initialization, the autoencoder is built, and the clustering mechanisms are all set into place for easy prediction. By default, the encoder takes on a structure (dimensions of data, 500, 500, 2000, ndim), while the decoder takes on the mirror of that structure. To alter the structure, we can adjust the architecture component when we initialize. ::
        
        harcluster_new_arch = n2d.n2d(x, manifoldGMM, ndim = n_clusters, architecture = [500, 2000, 500, 100])


In this case, the encoder part of the autoencoder would have structure (dimensions of data, 500, 2000, 500, 100, ndim). Please note that the autoencoder design defaults are sane, based on academic research, and produce excellent results, so the architecture does not require a lot of change in general. 

**Important Note**
In general, it is a good idea to say that **ndim = n_clusters**, that is to say we want to reduce our data's dimensionality from whatever space it lies in to the same number of dimensions as we have clusters. However, it is important to think critically! If you have data with 5000+ features, and want to put it into 2 or 3 groups, you probably should not set ndim to be 2 or 3. That is expecting a ridiculous amount of your computer!!
You are in essence learning a function that will map any 5000 dimensional observation into 2 or 3 numbers. Intuitively, this is unrealistic. This will lead to a model which gets stuck after 150 epochs, and when you tell your colleagues about your issues you will get some very funny  looks!

Lets talk about the default arguments for the n2d initialization method:


.. list-table:: n2d init Arguments
        :widths: 25 25 25
        :header-rows: 1

        * - Argument
          - Default
          - Description
        * - x
          - no default
          - The data
        * - manifoldLearner
          - no default, best to use UmapGMM
          - The manifold learning and clustering mechanism
        * - autoencoder
          - the default N2D AutoEncoder class
          - The class of autoencoder you wish to use. Note this argument is ust a class
        * - architecture
          - [500, 500, 2000]
          - The layout of the hidden layers in the network, presented in list form
        * - ndim
          - 10
          - Number of dimensions you wish to represent the data in with the autoencoder
        * - ae_args
          - {"act":"relu"}
          - dictionary of extra arguments to pass into the autoencoder




Learning an Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to train the autoencoder to learn the embedding. This step is pretty easy. As this is our first run of the autoencoder, the only thing we need to input is the name we would like the weights to be stored under, as well as create a weights directory. ::
        

        harcluster.fit(weight_id = "weights/har-1000-ae_weights.h5")

This will train the autoencoder, and store the weights in **weights/[WEIGHT_ID]-[NUM_EPOCHS]-ae_weights.h5**. The arguments to the preTrainEncoder method are shown in the table below:

.. list-table:: fit Arguments
        :widths: 25 25 25
        :header-rows: 1

        * - Argument
          - Default
          - Description
        * - batch_size
          - 256
          - The batch size
        * - pretrain_epochs
          - 1000
          - number of epochs
        * - loss
          - "mse"
          - The loss function
        * - optimizer
          - "adam"
          - The optimizier
        * - weights
          - None
          - The name of the weight file
        * - verbose
          - 0
          - The verbosity of the training
        * - weight_id
          - 'generic_autoencoder'
          - The name of the autoencoder used to identify the weights
        * - patience
          - None
          - int or None. If None, nothing special happens, if int, the tolerance for early stopping

Please note the patience parameter! It can save lots of time. A generally sane value for patience is 5. If after 5 epochs, loss does not decrease, the model will automatically stop for you!

On our next round of the autoencoder, while we fiddle with clustering algorithms, visualizations, or whatever, we can use the preTrainEncoder method to load in our weights as follows. ::
        
        harcluster.fit(weights = "weights/har-1000-ae_weights.h5")




Finally, we can actually cluster the data! To do this, we pass the clustering mechanism into the N2D predict method. ::
        
        harcluster.predict()

By default, the dataset that was fit is clustered. By specifying **x = ...**, we can predict on new data.

This clusters the data and stores the predictions in ::

        harcluster.preds


Assessing and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To assess the quality of the clusters, you can A) use some custom assessment method on the predictions or B) if you have labels run ::
        
        harcluster.assess(y)
        # (0.81212, 0.71669, 0.64013) 

This prints out the cluster accuracy, NMI, and ARI metrics for our clusters. These values are top of the line for all clustering models on HAR. 


To visualize, we again have a built in method as well as tools for creating your own visualizations: 

**Built in**::

        harcluster.visualize(y, y_names, savePath = "viz/har", nclust = n_clusters)

**Custom** :

We need a few things for a visualization: The embedding and the the predictions. The embedding is stored in ::
        
        harcluster.hle

You typically want to plot the embedding as x and the clusters as y! Lets also check out what our clusters look like!


.. image:: ../examples/viz/har-n2d-predicted.png
        :width: 800px
        :height: 600px
        :scale: 100 %
        :alt: Predicted clusters
        :align: center

These are the predicted clusters, now lets look at the real groupings!

.. image:: ../examples/viz/har-n2d.png
        :width: 800px
        :height: 600px
        :scale: 100 %
        :alt: Actual groupings
        :align: center


Looks like we did a pretty good job!! One very interesting thing to note, is even though it got some things wrong, where it got them wrong is still useful. The stationary activities are all near each other, while the active activities are all together. N2D, with no features and labels, not only found useful clusters, but ones that provide real world intuition! This is a very powerful result.

Usage as a Fully Online Model
---------------------------------

Once the weights have been initialized, we can use an N2D object in a fully online manner, as it is unsupervised learning. This means, if we have some new data, **x_new**, we can just predict using that ::

        harcluster.predict(x_new)


This will use the autoencoder to map the data into the proper number of dimensions, and then learn the manifold and cluster that with the new data!

