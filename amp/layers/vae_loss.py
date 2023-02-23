from keras import backend
from keras import layers
from keras import metrics as keras_metrics
import tensorflow as tf


class VAELoss(layers.Layer):

    def __init__(
            self,
            rcl_weight: int,
            **kwargs):
        self.rcl_weight = rcl_weight
        super(VAELoss, self).__init__(**kwargs)

    def calculate_loss(
            self,
            x,
            generated_x
    ):
        rcl = self.rcl_weight * backend.mean(
            keras_metrics.sparse_categorical_crossentropy(x, generated_x), axis=-1)
        return rcl

    def call(self, inputs):
        x = inputs[0]
        generated_x = inputs[1]
        loss = self.calculate_loss(x, generated_x)
        # self.add_loss(loss, inputs=inputs)
        print(loss)
        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
