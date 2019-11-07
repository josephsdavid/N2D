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



In this example, we are going to use the most difficult dataset, fashion ::

        x, y, y_names = data.load_fashion


Building the model
---------------------
