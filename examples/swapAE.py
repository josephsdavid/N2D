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
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import sys
import umap
from keras.layers import Dense, Input
from keras.models import Model

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


n_clusters = 10

n2d.n2d
model = n2d.n2d(x.shape[-1], manifold_learner=n2d.UmapGMM(n_clusters),
                autoencoder = denoisingAutoEncoder, ae_dim = n_clusters,
                ae_args={'noise_factor': 0.5})

model.fit(x, weights="weights/fashion_denoise-1000-ae_weights.h5")

model.predict(x)

model.visualize(y, y_names,  n_clusters = n_clusters)
plt.show()
print(model.assess(y))
