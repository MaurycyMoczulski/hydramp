import keras
import matplotlib.pyplot as plt
import tensorflow as tf

from amp.models.new_layers.new_layers import MaskLayer, PCATransformLayer
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from keras import layers

# GET MODELS
from model_sample_eval.functions import perform_pca, sigmoid

serializer = BasicModelSerializer()
hydramp = serializer.load_model("models/final_models/HydrAMP/direction")
amp_classifier = serializer.load_model('models/amp_classifier')
amp_classifier_model = amp_classifier()


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
negative_encoded_np = encoded_np[amp_y == 0][5]


# DEFINE SAMPLING MODEL
encoded_input = layers.Input(shape=(64,))
expanded = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(encoded_input)
amp_conv_scores = hydramp.output_layer.conv1(expanded)
# TODO amp_pow_scores = layers.Lambda(lambda x: K.pow(x, 2))
amp_conv_scores_mask = MaskLayer([3], 4)(amp_conv_scores)
amp_latent_pred = layers.GlobalMaxPooling1D(
    data_format='channels_last')(amp_conv_scores_mask)
latent_pred_model = keras.Model(
    encoded_input, [amp_latent_pred, amp_conv_scores_mask])

decoded = hydramp.decoder(encoded_input)

decoded_input = layers.Input(shape=(25,))
amp_classifier_pred = amp_classifier(decoded_input)


# SAMPLE
epochs = 500
max_pred = .0
new_encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)
encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)

freq = 25
rows = epochs // freq
fig, axs = plt.subplots(rows, 1)
fig.set_figheight(4 * rows)
img_no = 0

path = []

lr = 1e-1
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
    noise = np.random.normal(1, 0.05, size=(len(grads),))
    new_encoded_tf += np.multiply(grads, noise) * lr

    path.append(new_encoded_tf[0])
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
        pca, pca_path, pca_loc, pca_kernels = perform_pca(hydramp, path_stacked)
        pca_original = pca.transform([negative_encoded_np])
        pca_samples = pca.transform(encoded_np)

        axs[img_no].scatter(
            pca_samples[:, 0][amp_y == 0],
            pca_samples[:, 1][amp_y == 0], alpha=.5, label='neg')
        axs[img_no].scatter(
            pca_samples[:, 0][amp_y == 1],
            pca_samples[:, 1][amp_y == 1], alpha=.1, label='pos')
        axs[img_no].scatter(
            pca_loc[:, 0], pca_loc[:, 1], label='mu')
        axs[img_no].scatter(
            pca_original[:, 0],
            pca_original[:, 1], label='original')
        axs[img_no].plot(
            pca_path[:, 0],
            pca_path[:, 1], label='new')
        axs[img_no].scatter(
            pca_kernels[:, 0],
            pca_kernels[:, 1], label='directions', alpha=1)

        for j in range(len(pca_kernels)):
            axs[img_no].arrow(
                0, 0, pca_kernels[j, 0],
                pca_kernels[j, 1], width=.0001)
        axs[img_no].set_title(
            f'PATH')
        axs[img_no].legend()

        img_no += 1

fig.savefig('model_sample_eval/sampling_process.pdf')
