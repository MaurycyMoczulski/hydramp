import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf


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


def perform_pca(hydramp, X):
    # X_lens = np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1))
    # X = X / X_lens
    kernels = get_kernels(hydramp)
    pca = PCA(n_components=8)
    X = pca.fit_transform(X)
    loc = hydramp.mvn.mixture.components_distribution.loc.numpy()
    loc_lens = np.reshape(np.linalg.norm(loc, axis=1), (loc.shape[0], 1))
    # loc = loc / loc_lens
    pca_loc = pca.transform(loc)
    pca_kernels = pca.transform(kernels)
    return pca, X, pca_loc, pca_kernels, kernels, loc


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def translate_peptide(encoded_peptide):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([alphabet[el-1] if el != 0 else "" for el in encoded_peptide])


def get_numeric_peptide(peptide_str):
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    pep = np.array([np.where(alphabet == el)[0].item() + 1 for el in peptide_str])
    full_peptide = np.zeros((1, 25))
    full_peptide[0, :len(pep)] = pep
    return full_peptide
