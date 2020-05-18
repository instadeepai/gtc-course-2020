import numpy as np
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, KLCoeffMixin, \
    PPOLoss, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable


tf = try_import_tf()

OTHER_AGENTS = "other_agents"


class CentralizedValueMixin(object):
    """Add methods to evaluate the central value function from the model"""
    def __init__(self):
        self.central_value_function = self.model.central_value_function(
            self.get_placeholder(SampleBatch.CUR_OBS),
            self.get_placeholder(OTHER_AGENTS)
        )

    def compute_central_vf(self, obs, other_agents):
        feed_dict = {
            self.get_placeholder(SampleBatch.CUR_OBS): obs,
            self.get_placeholder(OTHER_AGENTS): other_agents,
        }
        return self.get_session().run(self.central_value_function, feed_dict)


def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    """
    Grabs the opponent obs/act and includes it in the experience train_batch,
    and computes GAE using the central vf predictions
    """

    max_num_opponents = policy.model.max_num_opponents
    obs_space_shape = policy.model.obs_space_shape
    act_space_shape = policy.model.act_space_shape

    if policy.loss_initialized():
        # assert sample_batch["dones"][-1], \
        #    "Not implemented for train_batch_mode=truncate_episodes"
        assert other_agent_batches is not None

        # record the obs and actions of the other agents in the trajectory
        opponents = list(other_agent_batches.values())
        
        other_s_a = np.concatenate(
            [opp_batch[SampleBatch.CUR_OBS] for (_, opp_batch) in opponents] + 
            [one_hot_encoding(opp_batch[SampleBatch.ACTIONS], act_space_shape) for (_, opp_batch) in opponents],
            axis=-1,
        )
        pad_size = (max_num_opponents - len(other_agent_batches)) * (obs_space_shape + act_space_shape)
        sample_batch[OTHER_AGENTS] = np.pad(other_s_a, ((0, 0), (0, pad_size)), 'constant')
        
        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS],
            sample_batch[OTHER_AGENTS]
        )
    else:
        # policy hasn't initialized yet, use zeros for other agents' obs & and act placeholders
        other_obs = np.zeros_like(sample_batch[SampleBatch.CUR_OBS], dtype=np.float32)
        other_act = one_hot_encoding(np.zeros_like(sample_batch[SampleBatch.ACTIONS]), act_space_shape)

        sample_batch[OTHER_AGENTS] = np.concatenate(
            [other_obs for _ in range(max_num_opponents)]
            + [other_act for _ in range(max_num_opponents)],
            axis=-1
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

    train_batch = compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


def one_hot_encoding(values, n_classes):
    """Numpy implementation of one hot encoding"""
    return np.eye(n_classes, dtype=np.float32)[values]


def loss_with_central_critic(policy, model, dist_class, train_batch):
    """Copied from PPO but optimizing the central value function"""
    CentralizedValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.central_value_out = policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OTHER_AGENTS])

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_out,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    """Copied from PPO"""
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    """Report the explained variance of the central value function"""
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_out),
    }


CCPPO = PPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])

CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO)

