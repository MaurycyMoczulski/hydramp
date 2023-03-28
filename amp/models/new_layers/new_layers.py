from typing import Dict
import keras
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

    # TODO tutaj zamiast z_mean, z_sigma, moze dać z po losowaniu z noisa ?
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
        self.conv2 = conv2
        self.name = name

    def call(self, z):
        amp_output = keras.activations.sigmoid(self.max_layer(self.conv1(K.expand_dims(z, axis=-1)))[:, 0])
        mic_output = keras.activations.sigmoid(self.max_layer(self.conv2(K.expand_dims(z, axis=-1)))[:, 0])
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
