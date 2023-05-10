import matplotlib.pyplot as plt
import tensorflow as tf
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from sklearn.decomposition import PCA

serializer = BasicModelSerializer()
model = serializer.load_model("models/final_models/HydrAMP/9")

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
kernels_pca = pca.transform(kernels)

rows = 3
fig, axs = plt.subplots(rows, 2)
fig.set_figheight(4.5 * rows)
img_no = 0

axs[img_no // 2, img_no % 2].bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
img_no += 1

for i in range(4):
    best_pca = np.argsort(np.abs(kernels_pca[i]))[-2:]
    for k in range(len(best_pca) - 1):
        axs[img_no // 2, img_no % 2].scatter(pca_samples[:, best_pca[k]][amp_y == 0], pca_samples[:, best_pca[k+1]][amp_y == 0], alpha=0.3)
        axs[img_no // 2, img_no % 2].scatter(pca_samples[:, best_pca[k]][amp_y == 1], pca_samples[:, best_pca[k+1]][amp_y == 1], alpha=.05)
        axs[img_no // 2, img_no % 2].scatter(loc_pca[:, best_pca[k]], loc_pca[:, best_pca[k+1]])
        for j in range(len(kernels_pca)):
            axs[img_no // 2, img_no % 2].scatter(kernels_pca[j, best_pca[k]], kernels_pca[j, best_pca[k+1]], label=str(j))
            axs[img_no // 2, img_no % 2].arrow(0, 0, kernels_pca[j, best_pca[k]], kernels_pca[j, best_pca[k+1]], width=.0001)
        axs[img_no // 2, img_no % 2].set_title(f'Direction no {i} PCA ({best_pca[k]} x {best_pca[k+1]})')
        axs[img_no // 2, img_no % 2].legend()
        img_no += 1

fig.savefig('myplot.pdf')
