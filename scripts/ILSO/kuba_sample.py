import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from amp.utils.basic_model_serializer import BasicModelSerializer
import amp.data_utils.data_loader as data_loader
from amp.config import MIN_LENGTH, MAX_LENGTH
import keras.backend as K
import numpy as np
from keras import layers
from amp.utils.functions import perform_pca, levenshteinDistanceDP, translate_peptide, get_numeric_peptide
from amp.data_utils.sequence import pad, to_one_hot
from amp.config import hydra
import pandas as pd

model = "KubAMP_pure_extend"

serializer = BasicModelSerializer()
hydramp_model = serializer.load_model(f"../../models/ilso_models/{model}/2")
amp_classifier = serializer.load_model('../../models/amp_classifier')
amp_classifier_model = amp_classifier()

peptides_input = layers.Input(shape=(25,))
encoder = hydramp_model.encoder
encoder_model = hydramp_model.encoder(peptides_input)
encoded_input = layers.Input(shape=(64,))
decoder = hydramp_model.decoder
decoder_model = hydramp_model.decoder(encoded_input)
hydramp_model.loss_weights=hydra
hydramp_model = hydramp_model.build(input_shape=(MAX_LENGTH, 21))

def generate_peptides(num_peptide, epochs=20000):
    peptides_input = layers.Input(shape=(25,))

    negative_encoded_np = encoder_model.predict(num_peptide)[0]

    # DEFINE SAMPLING MODEL
    encoded_input = layers.Input(shape=(64,))
    normed = layers.Lambda(
        lambda x: x / K.expand_dims(tf.norm(x), axis=-1))(encoded_input)
    expanded_normed = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(normed)
    print(hydramp_model.get_layer("OutputLayer"))
    amp_conv_scores = hydramp_model.get_layer("OutputLayer").conv1(expanded_normed)

    amp_latent_pred = layers.Lambda(lambda x: (x + 1) / 2)(amp_conv_scores)
    latent_pred_model = keras.Model(
        encoded_input, [amp_latent_pred, amp_conv_scores])

    decoded_input = layers.Input(shape=(25,))
    amp_classifier_pred = amp_classifier(decoded_input)


    # SAMPLE
    max_pred = .0
    new_encoded_tf = tf.convert_to_tensor([negative_encoded_np], dtype=tf.float32)
    start = negative_encoded_np
    encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)

    generated_peptides = []

    lr = 5e-1
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(epoch)
        amp_latent_pred_out, amp_conv_scores_out = latent_pred_model(new_encoded_tf)

        amp_latent_scores_val = K.eval(amp_conv_scores_out[0])
        decoded_val = decoder_model.predict(new_encoded_tf.numpy())
        amp_classifier_pred_val = amp_classifier_pred.predict(
            np.argmax(decoded_val, axis=-1)
        )[0, 0]

        grads = K.gradients(amp_latent_pred_out, [new_encoded_tf])
        grads = K.eval(grads)[0][0]
        noise = np.random.normal(1, 0.1, size=(len(grads),))
        new_encoded_tf += np.multiply(grads, noise) * lr

        max_pred = max(max_pred, amp_classifier_pred_val)

        decoded_peptide = np.argmax(decoded_val[0], axis=1)
        leven = levenshteinDistanceDP(decoded_peptide, num_peptide[0])
        translated_peptide = translate_peptide(decoded_peptide)
        generated_peptides.append(translated_peptide)

        if leven > 8:
            max_pred = .0
            noise = np.random.normal(1, .2, size=(len(grads),))
            start = np.multiply(negative_encoded_np, noise)
            new_encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)
            encoded_tf = tf.convert_to_tensor([start], dtype=tf.float32)
            
    generated_peptides = np.unique(generated_peptides)
    return generated_peptides

peptides = [
    "ILRWPWWPWRRK", # Omiganan (MBI-226)
    "AKRHHGYKRKFH", # Demegen (P-113)
    "GRRRRSVQWCA", # hLF1-11
    "KNKRKRRRRRRGGRRRR", # Mel4
    "MPKEKVFLKIEKMGRNIRN", # Vishnu3 (HB-107)
]
all_generated = []
for pep in peptides:
    num_pep = pad(to_one_hot([pep]))
    seqs = generate_peptides(num_pep, 1200)
    all_generated.extend(seqs)

# np.savetxt(f"../../results/ilso_results/{model}.txt", np.array(all_analogues), delimiter=",")
pd.DataFrame({"sequence": all_generated}).to_csv(f"../../results/ilso_results/{model}.txt")