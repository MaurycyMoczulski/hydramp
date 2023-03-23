import numpy as np

from amp.utils.basic_model_serializer import BasicModelSerializer
import tensorflow_probability as tfp

serializer = BasicModelSerializer()
model = serializer.load_model("models/final_models/HydrAMP/0")

# model.mvn.mixture.sample()
kernel = np.zeros_like(model.mvn.mixture.components_distribution.loc[0])
kernel[:len(model.output_layer.conv1.loaded_weights[0])] = model.output_layer.conv1.loaded_weights[0].flatten()
# print(model.mvn.mixture.components_distribution.scale.diag.shape)
# print(model.mvn.mixture.components_distribution.loc.shape)
# print(model.mvn.mixture.mixture_distribution.logits)

distribution_nr = model.mvn.mixture.mixture_distribution.sample().numpy()

mvn = tfp.distributions.MultivariateNormalDiag(
    loc=kernel - model.mvn.mixture.components_distribution.loc[distribution_nr],
    scale_diag=np.full_like(kernel, fill_value=0.05),
)

mvn.sample()

