from . import N2D
from tensorflow.keras.models import Model
import numpy as np

class manifold_cluster_generator(N2D.UmapGMM):
    def __init__(self, manifold_class, manifold_args, cluster_class, cluster_args, predict_method = "default"):
        # cluster exceptions
        self.predict_method = predict_method
        self.manifold_in_embedding = manifold_class(**manifold_args)
        self.cluster_manifold = cluster_class(**cluster_args)
        proba = getattr(self.cluster_manifold, "predict_proba", None)
        self.proba = callable(proba)
        self.hle = None

    def fit(self, hl):
        super().fit(hl)

    def predict(self, hl):
        if self.predict_method == "hdbscan-labels":
            return self.cluster_manifold.labels_
        elif self.predict_method == "hdbscan-prob":
            return self.cluster_manifold.probabilities_
        else:
            if self.proba:
                super().predict(hl)
            else:
                manifold = self.manifold_in_embedding.transform(hl)
                y_pred = self.cluster_manifold.predict(manifold)
                return(np.asarray(y_pred))

    def fit_predict(self, hl):
        if self.proba:
            super().fit_predict(hl)
        else:
            self.hle = self.manifold_in_embedding.fit_transform(hl)
            y_pred = self.cluster_manifold.fit_predict(self.hle)
            return(np.asarray(y_pred))


class autoencoder_generator(N2D.AutoEncoder):
    def __init__(self, model_levels=(), x_lambda = lambda x: x):
        self.Model = Model(model_levels[0], model_levels[2])
        self.encoder = Model(model_levels[0], model_levels[1])
        self.x_lambda = x_lambda

    def fit(self, x, batch_size, epochs,
            loss, optimizer, weights,
            verbose, weight_id, patience):
        super().fit(x, batch_size, epochs,
            loss, optimizer, weights,
            verbose, weight_id, patience)

