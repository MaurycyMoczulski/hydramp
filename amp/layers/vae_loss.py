import numpy as np
from keras import backend
from keras import layers
from keras import metrics as keras_metrics
import tensorflow as tf


class VAELoss(layers.Layer):

    def __init__(
            self,
            kl_weight: tf.Variable,
            rcl_weight: int,
            k_dim: int,
            **kwargs):
        self.kl_weight = kl_weight
        self.rcl_weight = rcl_weight
        self.gauss_variables = tf.random.uniform((k_dim, 2))
        if k_dim == 1:
            self.gauss_variables[0] = 0
            self.gauss_variables[1] = 1
        super(VAELoss, self).__init__(**kwargs)

    def calculate_loss(
            self,
            x,
            generated_x,
            z_sigma,
            z_mean,
            k,
    ):
        rcl = self.rcl_weight * backend.mean(
            keras_metrics.sparse_categorical_crossentropy(x, generated_x), axis=-1)
        kl_loss = 0
        for i in range(len(self.gauss_variables)):
            kl_loss += k[:, i] - 0.5 * backend.sum(
                1 + z_sigma / self.gauss_variables[i, 1] / -
                backend.square(z_mean - self.gauss_variables[i, 0]) * self.gauss_variables[i, 1] -
                backend.exp(z_sigma / self.gauss_variables[i, 1]), axis=-1)
        return rcl + (self.kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        generated_x = inputs[1]
        z_sigma = inputs[2]
        z_mean = inputs[3]
        k = inputs[4]
        loss = self.calculate_loss(x, generated_x, z_sigma, z_mean, k)
        #self.add_loss(loss, inputs=inputs)
        print(loss)
        return loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
