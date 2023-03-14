from typing import Any, Dict, Optional, List
import tensorflow as tf
from amp.layers import vae_loss
from amp.models import model as amp_model
from amp.models.decoders import amp_expanded_decoder
from amp.models.decoders import decoder as dec
from amp.models.discriminators import amp_classifier_noCONV
from amp.models.discriminators import discriminator as disc
from amp.models.discriminators import veltri_amp_classifier
from amp.models.encoders import amp_expanded_encoder
from amp.models.encoders import encoder as enc
from amp.utils import metrics
from keras import backend as K
from keras import layers, models, optimizers, losses
import keras
import tensorflow_probability as tfp


class OutputLayer(keras.layers.Layer):
    def __init__(self, kernel_size, latent_dim):
        super(OutputLayer, self).__init__()
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.max_layer = layers.GlobalMaxPooling1D(data_format='channels_first')
        self.conv1 = layers.Conv1D(1, self.kernel_size, strides=int(self.kernel_size / 4), activation='sigmoid')
        self.conv2 = layers.Conv1D(1, self.kernel_size, strides=int(self.kernel_size / 4), activation='sigmoid')
        self.conv1.trainable = False
        self.conv2.trainable = False
        super(OutputLayer, self).build(input_shape)

    def call(self, z):
        amp_output = self.max_layer(self.conv1(K.expand_dims(z, axis=-1)))
        mic_output = self.max_layer(self.conv2(K.expand_dims(z, axis=-1)))
        return tf.concat([amp_output, mic_output], axis=-1)

    def predict(self, z):
        return self(K.constant(z))


class MasterAMPTrainer(amp_model.Model):

    def __init__(
            self,
            encoder: enc.Encoder,
            decoder: dec.Decoder,
            amp_classifier: disc.Discriminator,
            mic_classifier: disc.Discriminator,
            kl_weight: float,
            rcl_weight: int,
            master_optimizer: optimizers.Optimizer,
            loss_weights: Optional[List[float]],
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.amp_classifier = amp_classifier
        self.mic_classifier = mic_classifier
        self.kl_weight = kl_weight
        self.rcl_weight = rcl_weight
        self.master_optimizer = master_optimizer
        self.loss_weights = loss_weights
        self.latent_dim = self.encoder.latent_dim
        self.kernel_size = int(self.latent_dim / 2)
        self.output_layer = OutputLayer(kernel_size=self.kernel_size, latent_dim=self.latent_dim)
        self.mvn = tfp.distributions.MultivariateNormalDiag(
            loc=K.random_normal((self.latent_dim,)),
            scale_diag=K.random_normal((self.latent_dim,))
        )

    @staticmethod
    def sampling(input_: Optional[Any] = None):
        noise_in, z_mean, z_sigma = input_
        return z_mean + K.exp(z_sigma / 2) * noise_in

    def build(self, input_shape: Optional):
        if self.loss_weights is None:
            raise AttributeError("Please set loss weight before training. Configs can be found at amp.config")
        self.amp_classifier.freeze_layers()
        self.mic_classifier.freeze_layers()

        sequences_input = layers.Input(shape=(input_shape[0],), name="sequences_input")
        z_mean, z_sigma, z = self.encoder.output_tensor(sequences_input)
        mic_in = layers.Input(shape=(1,), name="mic_in")
        amp_in = layers.Input(shape=(1,), name="amp_in")
        # noise_in is a noise applied to sampled z, must be defined as input to model
        noise_in = layers.Input(shape=(64,), name="noise_in")

        mvn = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_sigma)
        z = layers.Lambda(self.sampling, output_shape=(64,), name="z")
        z = z([noise_in, z_mean, z_sigma])
        reconstructed = self.decoder.output_tensor(z)
        z_reconstructed = self.encoder.output_tensor_with_dense_input(reconstructed)[0]
        z_reconstructed_error = layers.Subtract(name="z_reconstructed_error")([z, z_reconstructed])
        # end of cvae

        unconstrained_z = noise_in
        unconstrained_reconstructed = self.decoder.output_tensor(unconstrained_z)
        z_unconstrained_reconstructed = \
            self.encoder.output_tensor_with_dense_input(unconstrained_reconstructed)[0]
        unconstrained_reconstructed_error = layers.subtract(
            [noise_in, z_unconstrained_reconstructed], name="z_unconstrained_reconstructed_error")

        out = self.output_layer(z)
        reshape = layers.Reshape((1,))
        amp_output = layers.Lambda(lambda x: reshape(x[:, 0]))(out)
        mic_output = layers.Lambda(lambda x: reshape(x[:, 1]))(out)
        # CLASSIFIERS NAME wrappers
        # in order to appropriately name each output an identity layer lambda wrapper is added

        def idn_f(x_):
            return x_

        amp_output_wrap = \
            layers.Lambda(idn_f, name="amp_prediction")(amp_output)
        mic_output_wrap = \
            layers.Lambda(idn_f, name="mic_prediction")(mic_output)

        # GRADS ----------------------------------------------------------------------------------------------
        # Every value of target Sobolev grad must be provided as input to the graph because of Keras mechanics

        mic_mean_grad = K.gradients(
            loss=mic_output,
            variables=[z_mean]
        )[0]

        amp_mean_grad = K.gradients(
            loss=amp_output,
            variables=[z_mean]
        )[0]

        mic_mean_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, mic_mean_grad),
            name="mic_mean_grad"
        )

        amp_mean_grad_input = layers.Input(
            tensor=tf.math.scalar_mul(self.decoder.activation.temperature, amp_mean_grad),
            name="amp_mean_grad"
        )

        y = vae_loss.VAELoss(
            rcl_weight=self.rcl_weight,
        )([sequences_input, reconstructed, z_mean, z_sigma])
        y = layers.Subtract()([y, layers.Lambda(lambda x: self.kl_weight * (mvn.log_prob(x) + self.mvn.entropy()))(z)])

        vae = models.Model(
            inputs=[
                sequences_input,
                amp_in,
                mic_in,
                noise_in,
                mic_mean_grad_input,
                amp_mean_grad_input,
            ],
            outputs=[
                amp_output_wrap,
                mic_output_wrap,
                y,
                mic_mean_grad_input,
                amp_mean_grad_input,
                z_reconstructed_error,
                unconstrained_reconstructed_error,
            ]
        )

        kl_metric = - (mvn.log_prob(z) + self.mvn.entropy()) * self.kl_weight

        def _kl_metric(y_true, y_pred):
            return kl_metric

        reconstruction_acc = metrics.sparse_categorical_accuracy(sequences_input, reconstructed)

        def _reconstruction_acc(y_true, y_pred):
            return reconstruction_acc

        rcl = metrics.reconstruction_loss(sequences_input, reconstructed)

        def _rcl(y_true, y_pred):
            return rcl

        amino_acc, empty_acc = metrics.get_generation_acc()(sequences_input, reconstructed)

        def _amino_acc(y_true, y_pred):
            return amino_acc

        def _empty_acc(y_true, y_pred):
            return empty_acc

        def entropy(y_true, y_pred):
            return K.log(y_pred + K.epsilon()) * y_pred + K.log(1 - y_pred + K.epsilon()) * (1 - y_pred)

        def entropy_smoothed_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred) + 0.1 * entropy(y_true, y_pred)

        vae.compile(
            optimizer='adam',
            loss=[
                entropy_smoothed_loss,  # amp - classifier output
                'mae',  # mic - classifier output
                'mae',  # reconstruction
                losses.Huber(),  # mic_mean_grad_input
                losses.Huber(),  # amp_mean_grad_input
                'mse',  # z reconstructed error
                'mse',  # unconstrained reconstructed error
            ],
            loss_weights=self.loss_weights,
            metrics=[
                ['acc', 'binary_crossentropy'],  # amp - classifier output
                ['mae', 'binary_crossentropy'],  # mic - classifier output
                [_kl_metric, _rcl, _reconstruction_acc, _amino_acc, _empty_acc],  # reconstruction
                ['mse', losses.Huber()],  # mic_mean_grad_input
                ['mse', losses.Huber()],  # amp_mean_grad_input
                ['mse', 'mae'],  # z reconstructed error
                ['mse', 'mae'],  # unconstrained reconstructed error

            ]
        )
        return vae

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'encoder_config_dict': self.encoder.get_config_dict(),
            'decoder_config_dict': self.decoder.get_config_dict(),
            'amp_config_dict': self.amp_classifier.get_config_dict(),
            'mic_config_dict': self.mic_classifier.get_config_dict(),
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: amp_model.ModelLayerCollection,
    ) -> "MasterAMPTrainer":
        return cls(
            encoder=amp_expanded_encoder.AMPEncoder.from_config_dict_and_layer_collection(
                config_dict=config_dict['encoder_config_dict'],
                layer_collection=layer_collection,
            ),
            decoder=amp_expanded_decoder.AMPDecoder.from_config_dict_and_layer_collection(
                config_dict=config_dict['decoder_config_dict'],
                layer_collection=layer_collection,
            ),
            amp_classifier=amp_classifier_noCONV.NoConvAMPClassifier.from_config_dict_and_layer_collection(
                config_dict=config_dict['amp_config_dict'],
                layer_collection=layer_collection,
            ),
            mic_classifier=veltri_amp_classifier.VeltriAMPClassifier.from_config_dict_and_layer_collection(
                config_dict=config_dict['mic_config_dict'],
                layer_collection=layer_collection,
            ),

            kl_weight=K.variable(0.1),
            rcl_weight=32,
            master_optimizer=optimizers.Adam(lr=1e-3),
            loss_weights=None
        )

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        layers_with_names = {}
        for name, layer in self.encoder.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.decoder.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.amp_classifier.get_layers_with_names().items():
            layers_with_names[name] = layer
        for name, layer in self.mic_classifier.get_layers_with_names().items():
            layers_with_names[name] = layer

        return layers_with_names
