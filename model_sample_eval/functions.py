import numpy as np
from sklearn.decomposition import PCA


def get_kernels(hydramp):
    i = 0
    kernels = []
    while i + len(hydramp.output_layer.conv1.loaded_weights[0]) <=\
            len(hydramp.mvn.mixture.components_distribution.loc[0]):
        kernel = np.zeros_like(hydramp.mvn.mixture.components_distribution.loc[0])
        kernel[i:i + len(hydramp.output_layer.conv1.loaded_weights[0])] =\
            hydramp.output_layer.conv1.loaded_weights[0].flatten()
        kernels.append(kernel)
        i += hydramp.output_layer.conv1.strides[0]

    return np.stack(kernels)


def perform_pca(hydramp, samples):
    kernels = get_kernels(hydramp)
    pca = PCA(n_components=8)
    pca_samples = pca.fit_transform(samples)
    pca_loc = pca.transform(
        hydramp.mvn.mixture.components_distribution.loc.numpy())
    pca_kernels = pca.transform(kernels)
    return pca, pca_samples, pca_loc, pca_kernels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

