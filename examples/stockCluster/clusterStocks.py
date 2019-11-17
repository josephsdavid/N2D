from preprocess import scale
import n2d
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy as np


sp = scale("Data/stock_close.csv")
x = np.asarray(sp.values)
x = x.reshape(476, 1225, 1)

x.shape[0]
x.shape[1]

class LSTMAE:
    def __init__(self, data, ndim, architecture, act = 'relu'):
        # in this simple case, architecture is unused, but we keep for compat
        # with n2d
        dims = [data.shape[0]] +  [data.shape[1]] + [ndim]
        self.dims = dims
        self.act = act
        self.x = Input(shape = (self.dims[1], 1), name = 'input')
        self.h = self.x
        # if manually naming, just start at one. The naming convention is input
        # layer gets named input, encoders get named encoder_NUM where number
        # goes up from 1, and decoders do the same!
        self.h = LSTM(self.dims[2], activation = self.act, name = 'encoder_1')(self.h)
        self.h = RepeatVector(self.dims[1])(self.h)
        self.h = LSTM(1, return_sequences = True)(self.h)
        self.Model = Model(inputs = self.x, outputs = self.h)

    def fit(self, dataset, batch_size = 256, pretrain_epochs = 1000,
                     loss = 'mse', optimizer = 'adam',weights = None,
                     verbose = 0, weightname = 'fashion'):
        if weights == None:
            self.Model.compile(
                loss = loss, optimizer = optimizer
            )
            self.Model.fit(
                dataset, dataset,
                batch_size = batch_size,
                epochs = pretrain_epochs
            )

            self.Model.save_weights("weights/" + weightname + "-" +
                                    str(pretrain_epochs) +
                                    "-ae_weights.h5")
        else:
            self.Model.load_weights(weights)


#import os
#os.environ[CUDA_VISIBLE_DEVICES] = ""

n_clusters = 5

model = n2d.n2d(x, manifoldLearner=n2d.UmapGMM(n_clusters),
        autoencoder = LSTMAE,
        ndim = n_clusters, ae_args={'act':'relu'})

model.fit(weight_id = "lstm_sp_500", batch_size=25, pretrain_epochs=100)
