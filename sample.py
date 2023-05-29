import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from keras import layers
from model_sample_eval.functions import perform_pca, levenshteinDistanceDP, translate_peptide, get_numeric_peptide

# GET MODELS
serializer = BasicModelSerializer()
hydramp = serializer.load_model("models/final_models/HydrAMP/one_gaus_one_vector")
amp_classifier = serializer.load_model('models/amp_classifier')
amp_classifier_model = amp_classifier()

loc = hydramp.mvn.mixture.components_distribution.loc.numpy()


# GET SAMPLES REPRESENTATIONS
'''
peptydy z tabelki
ILRWPWWPWRRK
AKRHHGYKRKFH
GRRRRSVQWCA
KNKRKRRRRRRGGRRRR
MPKEKVFLKIEKMGRNIRN
'''

init_peptide = 'ILRWPWWPWRRK'

num_peptide = get_numeric_peptide(init_peptide)

data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

samples_np, amp_y = data_manager.get_merged_data()

peptides_input = layers.Input(shape=(25,))
encoded = hydramp.encoder(peptides_input)

encoded_np = encoded.predict(samples_np)
negative_encoded_np = encoded.predict(num_peptide)[0]

# DEFINE SAMPLING MODEL
encoded_input = layers.Input(shape=(64,))
normed = layers.Lambda(
    lambda x: x / K.expand_dims(tf.norm(x), axis=-1))(encoded_input)
expanded_normed = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(normed)
amp_conv_scores = hydramp.output_layer.conv1(expanded_normed)

amp_latent_pred = layers.Lambda(lambda x: (x + 1) / 2)(amp_conv_scores)
latent_pred_model = keras.Model(
    encoded_input, [amp_latent_pred, amp_conv_scores])

decoded = hydramp.decoder(encoded_input)

decoded_input = layers.Input(shape=(25,))
amp_classifier_pred = amp_classifier(decoded_input)


# SAMPLE
epochs = 200000
max_pred = .0
new_encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)
start = negative_encoded_np
encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)

rows = 20
fig, axs = plt.subplots(rows, 2)
fig.set_figheight(3.5 * rows)
img_no = 0

path = []
generated_peptides = []

lr = 5e-1
for epoch in range(epochs):
    amp_latent_pred_out, amp_conv_scores_out = latent_pred_model(new_encoded_tf)

    amp_latent_scores_val = K.eval(amp_conv_scores_out[0])
    decoded_val = decoded.predict(new_encoded_tf.numpy())
    amp_classifier_pred_val = amp_classifier_pred.predict(
        np.argmax(decoded_val, axis=-1)
    )[0, 0]

    grads = K.gradients(amp_latent_pred_out, [new_encoded_tf])
    grads = K.eval(grads)[0][0]
    noise = np.random.normal(1, 0.1, size=(len(grads),))
    new_encoded_tf += np.multiply(grads, noise) * lr

    path.append(new_encoded_tf[0])
    max_pred = max(max_pred, amp_classifier_pred_val)

    decoded_peptide = np.argmax(decoded_val[0], axis=1)
    leven = levenshteinDistanceDP(decoded_peptide, num_peptide[0])
    translated_peptide = translate_peptide(decoded_peptide)
    generated_peptides.append(translated_peptide)

    if leven > 8:
        print(
            f'Ep: %s   '
            f'Latent %.5f   '
            f'Levenshtein %.5f   '
            f'Cls %.5f   '
            f'MaxCLS %.2f' % (
                epoch,
                K.eval(amp_latent_pred_out[0])[0],
                leven,
                amp_classifier_pred_val,
                max_pred
            )
        )
        print(translated_peptide)
        if img_no < rows:
            path_stacked = np.stack(path)
            for ver in [0, 1]:
                best_pca = None
                pca_path = None
                pca_samples = None
                pca = None
                pca_loc = None
                pca_kernels = None
                if ver == 0:
                    pca, pca_path, pca_loc, pca_kernels, kernels, loc = perform_pca(hydramp, path_stacked)
                    pca_samples = pca.transform(encoded_np)
                    best_pca = [0, 1]
                elif ver == 1:
                    pca, pca_samples, pca_loc, pca_kernels, kernels, loc = perform_pca(hydramp, encoded_np)
                    pca_path = pca.transform(path_stacked)
                    best_pca = np.argsort(np.abs(pca_kernels[0]))[-2:][::-1]
                pca_original = pca.transform([start])
                axs[img_no, ver].scatter(
                    pca_samples[:, best_pca[0]][amp_y == 0],
                    pca_samples[:, best_pca[1]][amp_y == 0], alpha=.5, label='neg')
                axs[img_no, ver].scatter(
                    pca_samples[:, best_pca[0]][amp_y == 1],
                    pca_samples[:, best_pca[1]][amp_y == 1], alpha=.1, label='pos')
                axs[img_no, ver].scatter(
                    pca_loc[:, best_pca[0]], pca_loc[:, best_pca[1]], label='mu')
                axs[img_no, ver].scatter(
                    pca_original[:, best_pca[0]],
                    pca_original[:, best_pca[1]], label='original')
                axs[img_no, ver].plot(
                    pca_path[:, best_pca[0]],
                    pca_path[:, best_pca[1]], label='new')
                for j in range(len(pca_kernels)):
                    axs[img_no, ver].scatter(
                        pca_kernels[j, best_pca[0]],
                        pca_kernels[j, best_pca[1]], label='directions', alpha=1)
                    axs[img_no, ver].arrow(
                        0, 0, pca_kernels[j, best_pca[0]],
                        pca_kernels[j, best_pca[1]], width=.0001)
                axs[img_no, ver].set_title(
                    f'PATH')
                axs[img_no, ver].legend(framealpha=0.3, markerscale=.7)
                axs[img_no, ver].set_aspect('equal', adjustable='box')
            img_no += 1

        path = []
        max_pred = .0
        noise = np.random.normal(1, .2, size=(len(grads),))
        start = np.multiply(negative_encoded_np, noise)
        new_encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)
        encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)

fig.savefig(f'model_sample_eval/sampling_process_{init_peptide}.png')
generated_peptides = np.unique(generated_peptides)

with open(f'model_sample_eval/{init_peptide}_new.txt', 'w') as f:
    for pep in generated_peptides:
        f.write(f'{pep}\n')
