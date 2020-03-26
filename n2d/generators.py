# Third party modules
import numpy as np
from tensorflow.keras.models import Model

# Local modules
from . import N2D


class manifold_cluster_generator(N2D.UmapGMM):
    def __init__(self, manifold_class, manifold_args, cluster_class, cluster_args):
        # cluster exceptions
        self.manifold_in_embedding = manifold_class(**manifold_args)
        self.cluster_manifold = cluster_class(**cluster_args)
        proba = getattr(self.cluster_manifold, "predict_proba", None)
        self.proba = callable(proba)
        self.hle = None

    def fit(self, hl):
        super().fit(hl)

    def predict(self, hl):
        if self.proba:
            super().predict(hl)
        else:
            manifold = self.manifold_in_embedding.transform(hl)
            y_pred = self.cluster_manifold.predict(manifold)
            return np.asarray(y_pred)

    def fit_predict(self, hl):
        if self.proba:
            super().fit_predict(hl)
        else:
            self.hle = self.manifold_in_embedding.fit_transform(hl)
            y_pred = self.cluster_manifold.fit_predict(self.hle)
            return np.asarray(y_pred)

    def predict_proba(self, hl):
        if self.proba:
            super().predict_proba(hl)
        else:
            print("Your clusterer cannot predict probabilities")


class autoencoder_generator(N2D.AutoEncoder):
    def __init__(self, model_levels=(), x_lambda=lambda x: x):
        self.Model = Model(model_levels[0], model_levels[2])
        self.encoder = Model(model_levels[0], model_levels[1])
        self.x_lambda = x_lambda

    def fit(
        self,
        x,
        batch_size,
        epochs,
        loss,
        optimizer,
        weights,
        verbose,
        weight_id,
        patience,
    ):
        super().fit(
            x,
            batch_size,
            epochs,
            loss,
            optimizer,
            weights,
            verbose,
            weight_id,
            patience,
        )
