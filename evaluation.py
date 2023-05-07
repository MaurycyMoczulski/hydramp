import matplotlib.pyplot as plt
import tensorflow as tf
from amp.utils.basic_model_serializer import BasicModelSerializer
import tensorflow_probability as tfp
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from sklearn.decomposition import PCA

serializer = BasicModelSerializer()
model = serializer.load_model("models/final_models/HydrAMP/0")


def sample_from_model(model):
    # model.mvn.mixture.sample()
    # print(model.mvn.mixture.components_distribution.scale.diag.shape)
    # print(model.mvn.mixture.components_distribution.loc)
    # print(model.mvn.mixture.mixture_distribution.logits)

    distribution_nr = model.mvn.mixture.mixture_distribution.sample().numpy()
    loc = model.mvn.mixture.components_distribution.loc[distribution_nr]

    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=loc,
        scale_diag=np.full_like(kernel, fill_value=0.05),
    )

    sample = mvn.sample()
    return np.argmax(K.eval(model.decoder.output_tensor(tf.expand_dims(sample, 0))), -1)


data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y = data_manager.get_merged_data()
samples = []
i = 0
while i < len(amp_x):
    samples.append(
        model.encoder.output_tensor(
            tf.convert_to_tensor(amp_x[i: i+5000], dtype=tf.float32)
        )[0]
    )
    i += 5000

samples = K.eval(tf.concat(samples, axis=0))

pca = PCA(n_components=8)
pca_samples = pca.fit_transform(samples)

loc_pca = pca.transform(model.mvn.mixture.components_distribution.loc.numpy())

i = 0
kernels = []
while i + len(model.output_layer.conv1.loaded_weights[0]) <=\
        len(model.mvn.mixture.components_distribution.loc[0]):
    kernel = np.zeros_like(model.mvn.mixture.components_distribution.loc[0])
    kernel[i:i + len(model.output_layer.conv1.loaded_weights[0])] =\
        model.output_layer.conv1.loaded_weights[0].flatten()
    kernels.append(kernel)
    i += model.output_layer.conv1.strides[0]

kernels = np.stack(kernels)
print(kernels.shape)
kernels_pca = pca.transform(kernels)

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()

plt.scatter(pca_samples[:, 0][amp_y == 0], pca_samples[:, 1][amp_y == 0], alpha=0.3)
plt.scatter(pca_samples[:, 0][amp_y == 1], pca_samples[:, 1][amp_y == 1], alpha=.05)
plt.scatter(loc_pca[:, 0], loc_pca[:, 1])
plt.scatter(kernels_pca[:, 0], kernels_pca[:, 1])
plt.show()

print(samples.mean(axis=0), samples.std(axis=0))
print(kernels.mean(axis=0), kernels.std(axis=0))
print()
