import matplotlib.pyplot as plt
from model_sample_eval.functions import perform_pca
from keras import layers
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import numpy as np

# GET MODEL
serializer = BasicModelSerializer()
hydramp = serializer.load_model("models/final_models/HydrAMP/9")


# GET SAMPLES REPRESENTATIONS
data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

samples_np, amp_y = data_manager.get_merged_data()

negative_peptides_input = layers.Input(shape=(25,))
encoded = hydramp.encoder(negative_peptides_input)

samples = encoded.predict(samples_np)

pca, pca_samples, pca_loc, pca_kernels, kernels, loc = perform_pca(hydramp, samples)

rows = 1 + 4 * len(kernels)
fig, axs = plt.subplots(rows, 1)
fig.set_figheight(4 * rows)
img_no = 0

axs[img_no].bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
img_no += 1

for i in range(len(kernels)):
    best_pca = np.argsort(np.abs(pca_kernels[i]))[-5:][::-1]
    for k in range(len(best_pca) - 1):
        axs[img_no].scatter(
            pca_samples[:, best_pca[k]][amp_y == 0],
            pca_samples[:, best_pca[k+1]][amp_y == 0], alpha=.5, label='neg')
        axs[img_no].scatter(
            pca_samples[:, best_pca[k]][amp_y == 1],
            pca_samples[:, best_pca[k+1]][amp_y == 1], alpha=.05, label='pos')
        axs[img_no].scatter(
            pca_loc[:, best_pca[k]], pca_loc[:, best_pca[k+1]], label='loc')
        for j in range(len(kernels)):
            axs[img_no].scatter(
                pca_kernels[j, best_pca[k]],
                pca_kernels[j, best_pca[k+1]], label=str(j))
            axs[img_no].arrow(
                0, 0, pca_kernels[j, best_pca[k]], pca_kernels[j, best_pca[k+1]], width=.0001)
        axs[img_no].set_title(
            f'Direction no {i} PCA ({best_pca[k]} x {best_pca[k+1]})')
        axs[img_no].legend(framealpha=0.3, markerscale=.7)
        axs[img_no].set_aspect('equal', adjustable='box')
        img_no += 1

fig.savefig('model_sample_eval/latent2.pdf')
