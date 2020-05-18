from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input
from .dense_model_tf import DenseNetwork


tf = try_import_tf()


class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        
        self.params = model_config["custom_options"]
        self.max_num_agents = self.params["max_num_agents"]
        self.max_num_opponents = self.max_num_agents - 1
        self.obs_space_shape = obs_space.shape[0]
        self.act_space_shape = action_space.n

        # Actor Layers
        actor_inputs = Input(shape=(self.obs_space_shape,), name="a_obs")
        self.actor = DenseNetwork(num_outputs=self.act_space_shape, name="Actor", **self.params["actor"])
        action_logits = self.actor(actor_inputs)

        # Actor Model
        self.actor_model = Model(actor_inputs, action_logits)
        self.register_variables(self.actor_model.variables)

        # Critic Layers
        obs = Input(shape=(self.obs_space_shape,), name="c_obs")
        opp_obs_act = Input(shape=((self.obs_space_shape + self.act_space_shape) * self.max_num_opponents,), name="opp_obs_act")
        concat_obs = Concatenate(axis=1)([obs, opp_obs_act])
        self.central_critic = DenseNetwork(num_outputs=1, name="Critic", **self.params["critic"])
        central_vf = self.central_critic(concat_obs)

        # Critic Model
        self.central_critic_model = Model([obs, opp_obs_act], central_vf)
        self.register_variables(self.central_critic_model.variables)

    def forward(self, input_dict, state, seq_lens):
        policy = self.actor_model(input_dict["obs_flat"])
        self._value_out = tf.reduce_mean(policy, axis=-1)
        return policy, state

    def central_value_function(self, obs, other_agent):
        """Maps (obs, opp_ops, opp_act) -> vf_pred"""
        return tf.reshape(self.central_critic_model([obs, other_agent]), [-1])

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
