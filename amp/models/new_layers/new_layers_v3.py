from typing import Dict
import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
from keras import layers
from amp.models import model


class VAEGMMLayer(layers.Layer):

    @property
    def name(self):
        return self._name

    def __init__(
            self,
            components_logits_np,
            component_means_np,
            component_diags_np,
            name: str = 'VAEGMMLayer'
    ):
        super(VAEGMMLayer, self).__init__()
        self.name = name
        self.components_logits_np = components_logits_np
        self.component_means_np = component_means_np
        self.component_diags_np = component_diags_np

        components_logits = K.variable(
            components_logits_np,
            name='gmm_logits'
        )
        component_means = K.variable(self.component_means_np)
        component_diags = K.constant(self.component_diags_np)

        self.mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=components_logits),
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=component_means,
                scale_diag=component_diags,
            )
        )

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'components_logits_np': self.components_logits_np,
            'component_means_np': self.component_means_np,
            'component_diags_np': self.component_diags_np
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {}

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "VAEGMMLayer":
        return cls(
            name=config_dict['name'],
            components_logits_np=config_dict['components_logits_np'],
            component_means_np=config_dict['component_means_np'],
            component_diags_np=config_dict['component_diags_np']
        )

    def call(self, x):
        posterior_loc, posterior_scale_diag = x[0], x[1]
        posterior_normal = tfp.distributions.MultivariateNormalDiag(
            loc=posterior_loc,
            scale_diag=posterior_scale_diag,
        )
        sample = posterior_normal.sample()

        prior_likelihood = self.mixture.log_prob(sample)
        posterior_likelihood = posterior_normal.log_prob(sample)

        sample_log_prior = prior_likelihood
        sample_log_posterior = posterior_likelihood
        return sample_log_prior + sample_log_posterior

    @name.setter
    def name(self, value):
        self._name = value


class OutputLayer(keras.layers.Layer):
    @property
    def name(self):
        return self._name

    def __init__(self, max_layer, conv1, conv2, name: str = 'OutputLayer'):
        super(OutputLayer, self).__init__()
        self.max_layer = max_layer
        self.conv1 = conv1
        self.name = name

    def call(self, z):
        self.conv1.weight = self.conv1.weight / tf.norm(self.conv1.weight)
        z_len = tf.norm(z)
        z_amp = layers.Lambda(lambda x: z / z_len)
        amp_output = keras.activations.sigmoid(self.max_layer(
            K.pow(self.conv1(K.expand_dims(z_amp, axis=-1)), 2))[:, 0])
        mic_output = z_len
        return tf.stack([amp_output, mic_output], axis=0)

    def predict(self, z):
        return self(K.constant(z))

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name
        }

    @name.setter
    def name(self, value):
        self._name = value

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {
            f'{self.name}_max_layer': self.max_layer,
            f'{self.name}_conv1': self.conv1,
            f'{self.name}_conv2': self.conv2,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "OutputLayer":
        return cls(
            name=config_dict['name'],
            max_layer=layer_collection[config_dict['name'] + '_max_layer'],
            conv1=layer_collection[config_dict['name'] + '_conv1'],
            conv2=layer_collection[config_dict['name'] + '_conv2'],
        )


class MaskLayer(keras.layers.Layer):
    @property
    def name(self):
        return self._name

    def __init__(self, idx, k, name: str = 'MaskLayer'):
        super(MaskLayer, self).__init__()
        mask = np.zeros(k)
        mask[idx] = 1
        self.mask = np.where(mask)[0]
        self.name = name

    def call(self, z):
        return tf.gather(z, self.mask, axis=1)

    def predict(self, z):
        return self(K.constant(z))

    @name.setter
    def name(self, value):
        self._name = value


class PCATransformLayer(keras.layers.Layer):
    @property
    def name(self):
        return self._name

    def __init__(self, mean, components_T, components_idx, n_components, name: str = 'PCATransformLayer'):
        super(PCATransformLayer, self).__init__()
        self.mean = K.variable(mean)
        self.components_T = K.variable(components_T)
        mask = np.ones(n_components)
        mask[components_idx] = 0
        self.mask = np.where(mask)[0]
        self.name = name

    def call(self, z):
        z = z - self.mean
        z_dot = K.dot(z, self.components_T)
        return tf.gather(z_dot, self.mask, axis=1)

    def predict(self, z):
        return self(K.constant(z))

    @name.setter
    def name(self, value):
        self._name = value
