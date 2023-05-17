import keras
import matplotlib.pyplot as plt
import tensorflow as tf

from amp.models.new_layers.new_layers_v3 import MaskLayer, PCATransformLayer
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from keras import layers

# GET MODELS
from model_sample_eval.functions import perform_pca, sigmoid, get_kernels

serializer = BasicModelSerializer()
hydramp = serializer.load_model("models/final_models/HydrAMP/9")
amp_classifier = serializer.load_model('models/amp_classifier')
amp_classifier_model = amp_classifier()

kernels = get_kernels(hydramp)

loc = hydramp.mvn.mixture.components_distribution.loc.numpy()
loc_lens = np.reshape(np.linalg.norm(loc, axis=1), (loc.shape[0], 1))
loc = loc / loc_lens

# GET SAMPLES REPRESENTATIONS
data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

samples_np, amp_y = data_manager.get_merged_data()

negative_peptides_input = layers.Input(shape=(25,))
encoded = hydramp.encoder(negative_peptides_input)

encoded_np = encoded.predict(samples_np)
negative_encoded_np = encoded_np[amp_y == 0][0]

encoded_np_len = np.reshape(np.linalg.norm(encoded_np, axis=1), (encoded_np.shape[0], 1))
encoded_np_norm = encoded_np / encoded_np_len

negative_encoded_np_len = np.linalg.norm(negative_encoded_np)
negative_encoded_np_norm = negative_encoded_np / negative_encoded_np_len


# DEFINE SAMPLING MODEL
encoded_input = layers.Input(shape=(64,))
normed = layers.Lambda(
    lambda x: x / K.expand_dims(tf.norm(x, axis=1), axis=-1))(encoded_input)
expanded_normed = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(normed)
amp_conv_scores = hydramp.output_layer.conv1(expanded_normed)
amp_conv_scores_mask = MaskLayer([0, 1, 2, 3], 4)(amp_conv_scores)
amp_latent_vector_dot = layers.GlobalMaxPooling1D(
    data_format='channels_last')(amp_conv_scores_mask)
amp_latent_pred = layers.Lambda(lambda x: (x + 1) / 2)(amp_latent_vector_dot)
latent_pred_model = keras.Model(
    encoded_input, [amp_latent_pred, amp_conv_scores_mask])

decoded = hydramp.decoder(encoded_input)

decoded_input = layers.Input(shape=(25,))
amp_classifier_pred = amp_classifier(decoded_input)


# SAMPLE
epochs = 150
max_pred = .0
new_encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)
encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)

freq = 25
rows = epochs // freq
fig, axs = plt.subplots(rows, 1)
fig.set_figheight(4 * rows)
img_no = 0

path = []

lr = 1
for epoch in range(epochs):
    amp_latent_pred_out, amp_conv_scores_out = latent_pred_model(new_encoded_tf)

    amp_latent_scores_val = K.eval(amp_conv_scores_out[0])
    decoded_val = decoded.predict(new_encoded_tf.numpy())
    amp_classifier_pred_val = amp_classifier_pred.predict(
        np.argmax(decoded_val, axis=-1)
    )[0, 0]

    mse_loss = keras.losses.MeanSquaredError()(
        encoded_tf, new_encoded_tf)

    grads = K.gradients(amp_latent_pred_out, [new_encoded_tf])
    grads = K.eval(grads)[0][0]
    noise = np.random.normal(1, 0, size=(len(grads),))
    new_encoded_tf += np.multiply(grads, noise) * lr

    new_encoded_tf_norm = new_encoded_tf / np.reshape(
        np.linalg.norm(new_encoded_tf, axis=1), (new_encoded_tf.shape[0], 1))

    path.append(new_encoded_tf_norm[0])
    max_pred = max(max_pred, amp_classifier_pred_val)

    print(
        f'Ep: %s   '
        f'Latent %.5f   '
        f'Mse %.5f   '
        f'Cls %.5f   '
        f'MaxCLS %.2f' % (
            epoch,
            sigmoid(K.eval(amp_latent_pred_out[0])[0]),
            K.eval(mse_loss),
            amp_classifier_pred_val,
            max_pred
        )
    )

    if epoch % freq == freq - 1:
        path_stacked = np.stack(path)
        '''
        path_stacked_len = np.reshape(
            np.linalg.norm(path_stacked, axis=1), (path_stacked.shape[0], 1))
        path_stacked_norm = path_stacked / path_stacked_len
        pca, pca_path, pca_loc, pca_kernels, kernels, loc = perform_pca(hydramp, path_stacked)
        pca_original = pca.transform([negative_encoded_np_norm])
        best_pca = [0, 1]
        pca_samples = pca.transform(encoded_np_norm)
        axs[img_no].scatter(
            pca_samples[:, best_pca[0]][amp_y == 0],
            pca_samples[:, best_pca[1]][amp_y == 0], alpha=.5, label='neg')
        axs[img_no].scatter(
            pca_samples[:, best_pca[0]][amp_y == 1],
            pca_samples[:, best_pca[1]][amp_y == 1], alpha=.1, label='pos')
        axs[img_no].scatter(
            pca_loc[:, best_pca[0]], pca_loc[:, best_pca[1]], label='mu')
        axs[img_no].scatter(
            pca_original[:, best_pca[0]],
            pca_original[:, best_pca[1]], label='original')
        axs[img_no].plot(
            pca_path[:, best_pca[0]],
            pca_path[:, best_pca[1]], label='new')
        for j in range(len(pca_kernels)):
            axs[img_no].scatter(
                pca_kernels[j, best_pca[0]],
                pca_kernels[j, best_pca[1]], label='directions', alpha=1)
            axs[img_no].arrow(
                0, 0, pca_kernels[j, best_pca[0]],
                pca_kernels[j, best_pca[1]], width=.0001)
        axs[img_no].set_title(
            f'PATH')
        axs[img_no].legend(framealpha=0.3, markerscale=.7)
        axs[img_no].set_aspect('equal', adjustable='box')
        img_no += 1
        '''
        t_matrix = np.zeros((64, 2))
        #t_matrix[:, 0] = (kernels[2] == 0).astype(float) / np.sqrt(48)
        #t_matrix[:, 1] = kernels[2]
        t_matrix[0, 0] = 1
        t_matrix[36, 1] = 1
        t_samples = np.matmul(encoded_np_norm, t_matrix)
        t_kernels = np.matmul(kernels, t_matrix)
        t_loc = np.matmul(loc, t_matrix)
        t_original = np.matmul([negative_encoded_np_norm], t_matrix)
        t_path = np.matmul(path_stacked, t_matrix)
        axs[img_no].scatter(
            t_samples[:, 0][amp_y == 0],
            t_samples[:, 1][amp_y == 0], alpha=.5, label='neg')
        axs[img_no].scatter(
            t_samples[:, 0][amp_y == 1],
            t_samples[:, 1][amp_y == 1], alpha=.05, label='pos')
        axs[img_no].scatter(
            t_loc[:, 0], t_loc[:, -1], label='loc')
        axs[img_no].scatter(
            0, t_kernels[2, 1], label=str(2))
        axs[img_no].arrow(
            0, 0, 0, t_kernels[2, 1], width=.0001)
        axs[img_no].set_title(
            f'Direction no {2}')
        axs[img_no].scatter(
            t_original[:, 0],
            t_original[:, 1], label='original')
        axs[img_no].plot(
            t_path[:, 0],
            t_path[:, 1], label='new')
        axs[img_no].legend()
        img_no += 1

fig.savefig('model_sample_eval/sampling_process.pdf')
