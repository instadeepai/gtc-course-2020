from ray.rllib.models.preprocessors import Preprocessor

from environments.obs_utils import normalize_observation


class TreeObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.tree_depth = options["custom_options"]["tree_depth"]
        self.observation_radius = options["custom_options"]["observation_radius"]
        return (obs_space.shape[0],)

    def transform(self, observation):
        return normalize_observation(
            observation=observation,
            tree_depth=self.tree_depth,
            observation_radius=self.observation_radius,
        )
