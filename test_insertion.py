from __future__ import annotations
import sys
import traceback
import gymnasium as gym
import numpy as np
import gym_inserting


def describe_space(space: gym.Space) -> str:
    if hasattr(space, "shape"):
        return f"{type(space).__name__}(shape={space.shape}, dtype={getattr(space, 'dtype', None)})"
    if hasattr(space, "n"):
        return f"{type(space).__name__}(n={space.n})"
    return type(space).__name__

def main() -> int:
    env_id = "gate_insertion-v0"
    env = gym.make(env_id, render=False, max_steps_per_episode=100)

    print("Created env:", env)
    print("  spec:", getattr(env, "spec", None))
    print("  metadata:", getattr(env, "metadata", None))

    env = env.unwrapped
    env.start()
        
    obs, info = env.reset()
    print("Reset done.", obs.shape if isinstance(obs, np.ndarray) else type(obs))
    
    def summarize_obs(o):
        if isinstance(o, np.ndarray):
            return f"ndarray(shape={o.shape}, dtype={o.dtype})"
        if isinstance(o, (list, tuple)):
            return f"{type(o).__name__}([" + ", ".join(summarize_obs(x) for x in o) + "])"
        return f"{type(o).__name__}: {o}"

    print("  obs summary:", summarize_obs(obs))
    print("  info keys:", list(info.keys()) if isinstance(info, dict) else type(info))

    action = np.random.uniform(low=-1.0, high=1.0, size=7) # j_pos
    print("Stepping once...", action)
    step_out = env.step(action)
    obs2, reward, terminated, truncated, info2 = step_out
    done = bool(terminated or truncated)
    print("Step complete:")
    print("  reward:", reward)
    print("  terminated:", terminated, " truncated:", truncated, " done:", done)
    print("  next obs summary:", summarize_obs(obs2))
    print("  info keys:", list(info2.keys()) if isinstance(info2, dict) else type(info2))

    env.close()
    print("Env closed. Sanity check completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
