import matplotlib.pyplot as plt
import numpy as np
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
    kernel = np.zeros_like(model.mvn.mixture.components_distribution.loc[0])
    kernel[:len(model.output_layer.conv1.loaded_weights[0])] = model.output_layer.conv1.loaded_weights[0].flatten()
    # print(model.mvn.mixture.components_distribution.scale.diag.shape)
    # print(model.mvn.mixture.components_distribution.loc.shape)
    # print(model.mvn.mixture.mixture_distribution.logits)

    distribution_nr = model.mvn.mixture.mixture_distribution.sample().numpy()
    loc = kernel - model.mvn.mixture.components_distribution.loc[distribution_nr]

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

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()

print(pca_samples.shape)

plt.scatter(pca_samples[:, 0][amp_y == 0], pca_samples[:, 0][amp_y == 0])
plt.scatter(pca_samples[:, 0][amp_y == 1], pca_samples[:, 0][amp_y == 1], alpha=.01)
plt.show()


