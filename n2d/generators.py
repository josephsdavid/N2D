import N2D

class manifold_cluster_generator(N2D.UmapGMM):
    def __init__(manifold_class, manifold_args, cluster_class, cluster_args):
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
            return(np.asarray(y_pred))

    def fit_predict(self, hl):
        if self.proba:
            super().fit_predic(hl)
        else:
            self.hle = self.manifold_in_embedding.fit_transform(hl)
            self.cluster_manifold.fit(self.hle)
            y_pred = self.cluster_manifold.predict(self.hle)
            return(np.asarray(y_pred))



