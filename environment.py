import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts

import logic
from game import step


class GameEnv(PyEnvironment):
    """
    Custom `PyEnvironment` environment.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self):
        """Initialization of environment with X_train and y_train."""
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name="action")
        self._observation_spec = ArraySpec(shape=(16,), dtype=np.int32, name="observation")
        self._episode_ended = False

        self.episode_step = 0  # Episode step, resets every episode
        self._state = logic.reset_game()

    def action_spec(self):
        """Definition of the actions."""
        return self._action_spec

    def observation_spec(self):
        """Definition of the observations."""
        return self._observation_spec

    def _reset(self):
        """Shuffles data and returns the first state to begin training on new episode."""
        self.episode_step = 0  # Reset episode step counter at the end of every episode
        self._state = logic.reset_game()
        self._episode_ended = False

        return ts.restart(self._state.ravel())

    def _step(self, action):
        """Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_rate` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_rate` depending on the current class.
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()
        self.episode_step += 1

        state, reward, terminal = step(self._state, action)
        self._episode_ended = terminal  # Stop episode when minority class is misclassified
        self._state = state  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state.ravel(), reward)
        else:
            return ts.transition(self._state.ravel(), reward)
