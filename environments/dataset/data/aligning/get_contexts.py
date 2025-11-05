import numpy as np
import random
import pickle

import gymnasium as gym
# ensure env package is importable for side-effect registration
from environments.d3il.envs.gym_aligning_env import gym_aligning as _ga  # noqa: F401


env = gym.make('aligning-v0')
env.start()


random.seed(0)
np.random.seed(0)

test_contexts = []
for i in range(60):

    context = env.manager.sample()
    test_contexts.append(context)

with open("test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)


# file_lists = os.listdir("all_data")
file_lists = np.load("train_files.pkl", allow_pickle=True)

train_contexts = []

for file in file_lists[:60]:

    arr = np.load("all_data/state/" + file, allow_pickle=True,)

    train_contexts.append(arr["context"])

with open("train_contexts.pkl", "wb") as f:
    pickle.dump(train_contexts, f)