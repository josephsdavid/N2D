# Third party modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# First party modules
import n2d
from n2d import datasets as data

plt.style.use(["seaborn-white", "seaborn-paper"])
sns.set_context("paper", font_scale=1.3)

# load up data
x, y, y_names = data.load_fashion()

# define number of clusters
n_clusters = 10

# set up manifold learner
umapgmm = n2d.UmapGMM(n_clusters)

# set up parameters for denoising autoencoder
def add_noise(x, noise_factor):
    x_clean = x
    x_noisy = x_clean + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_clean.shape
    )
    x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return x_noisy


# define stages of networks
hidden_dims = [500, 500, 2000]
input_dim = x.shape[-1]
inputs = Input(input_dim)
encoded = inputs
for d in hidden_dims:
    encoded = Dense(d, activation="relu")(encoded)
encoded = Dense(n_clusters)(encoded)
decoded = encoded
for d in hidden_dims[::-1]:
    decoded = Dense(d, activation="relu")(decoded)
outputs = Dense(input_dim)(decoded)


# inputs: iterable of inputs, center, outputs of ae, lambda for noise
denoising_ae = n2d.autoencoder_generator(
    (inputs, encoded, outputs), x_lambda=lambda x: add_noise(x, 0.5)
)

# define model
model = n2d.n2d(denousing_ae, umapgmm)

# fit the model
model.fit(x, epochs=10)

# make some predictions
model.predict(x)

model.visualize(y, y_names, n_clusters=n_clusters)
plt.show()
print(model.assess(y))
