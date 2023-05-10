import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from amp.utils import basic_model_serializer

from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from sklearn.decomposition import PCA
from keras import layers, models, optimizers

# GET MODELS
serializer = BasicModelSerializer()
hydramp = serializer.load_model("models/final_models/HydrAMP/9")

bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('models/amp_classifier')
amp_classifier_model = amp_classifier()


# GET NEGATIVE SAMPLES REPRESENTATIONS
data_manager = data_loader.AMPDataManager(
    'data/unlabelled_positive.csv',
    'data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

amp_x, amp_y = data_manager.get_merged_data()
negative_samples_np = amp_x[amp_y == 0]

negative_peptides_input = layers.Input(shape=(25,))
encoded = hydramp.encoder(negative_peptides_input)

encoded_np = encoded.predict(negative_samples_np)[[5]]

# DEFINE SAMPLING MODEL
encoded_input = layers.Input(shape=(64,))
expanded = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(encoded_input)
amp_conv_scores = hydramp.output_layer.conv1(expanded)
amp_latent_pred = layers.GlobalMaxPooling1D(data_format='channels_last')(amp_conv_scores)
latent_pred_model = keras.Model(encoded_input, amp_latent_pred)

decoded = hydramp.decoder(encoded_input)

decoded_input = layers.Input(shape=(25,))
amp_classifier_pred = amp_classifier(decoded_input)

# SAMPLE
lr = 1e-1
epochs = 500
max_pred = .0
encoded_tf = tf.convert_to_tensor(encoded_np, dtype=tf.float32)
for epoch in range(epochs):
    amp_latent_pred_value = latent_pred_model(encoded_tf)
    decoded_value = decoded.predict(encoded_np)
    amp_classifier_pred_value = amp_classifier_pred.predict(
        np.argmax(decoded_value, axis=-1)
    )[0, 0]
    grads = K.gradients(amp_latent_pred_value, [encoded_tf])
    grads = K.eval(grads)
    encoded_tf += grads[0] * lr
    encoded_tf += np.random.normal(0, .01, size=(len(grads[0]),)) * lr
    max_pred = max(max_pred, amp_classifier_pred_value)
    print(
        f'Ep: %s  '
        f'Hydramp %.5f   '
        f'Classifier %.5f   '
        f'Max Classifier %.2f' % (
            epoch,
            K.eval(amp_latent_pred_value[0])[0],
            amp_classifier_pred_value,
            max_pred
        )
    )
