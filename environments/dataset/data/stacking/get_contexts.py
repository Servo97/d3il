import numpy as np
import random
import pickle
import gymnasium as gym
from environments.d3il.envs.gym_stacking_env import gym_stacking as _gs  # noqa: F401

env = gym.make('stacking-v0', max_steps_per_episode=1000, render=False)

random.seed(0)
np.random.seed(0)

test_contexts = []
for i in range(100):

    context = env.manager.sample()
    test_contexts.append(context)

with open("test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)

