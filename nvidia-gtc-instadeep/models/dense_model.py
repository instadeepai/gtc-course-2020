from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from models.dense_model_tf import DenseNetwork

tf = try_import_tf()


class DenseModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DenseModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.params = model_config["custom_options"]
        self.inputs = Input(shape=obs_space.shape, name="observations")
        self.embedding = DenseNetwork(name="Embedding", **self.params["embedding"])
        self.actor = DenseNetwork(num_outputs=num_outputs, name="Actor", **self.params["actor"])
        self.critic = DenseNetwork(num_outputs=1, name="Critic", **self.params["critic"])

        embedding = self.embedding(self.inputs)
        action_logits = self.actor(embedding)
        value = self.critic(embedding)

        self.base_model = Model(self.inputs, [action_logits, value])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return Reshape(())(self._value_out)
