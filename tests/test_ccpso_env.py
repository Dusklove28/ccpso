import numpy as np

from environments.ccpso_env import CCPSOEnv
from swarm.ccpso import CCPSOSwarm


def sphere(positions):
    return np.sum(positions ** 2, axis=1)


def main():
    swarm = CCPSOSwarm(
        particles=20,
        dimensions=10,
        fun=sphere,
        lower_bound=-100.0,
        upper_bound=100.0,
        max_fe=1000,
    )

    env = CCPSOEnv(
        swarm=swarm,
        conv_min=0.0,
        conv_max=1.5,
        optimum=0.0,
    )

    observation, info = env.reset(seed=42)

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert env.observation_space.contains(observation)

    assert env.action_space.shape == (1,)
    assert info["fe_count"] == 20

    episode_reward = 0.0
    step_count = 0

    while True:
        # action=0 应映射为 Conv_a=0.75
        action = np.array([0.0], dtype=np.float32)

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        assert next_observation.shape == (5,)
        assert next_observation.dtype == np.float32
        assert np.isfinite(next_observation).all()
        assert env.observation_space.contains(next_observation)

        assert np.isfinite(reward)
        assert 0.0 <= info["conv"] <= 1.5
        assert np.isclose(info["conv"], 0.75)
        assert info["fe_count"] <= 1000

        episode_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    assert swarm.fe_count == 1000
    assert truncated

    print("Environment reset: passed")
    print("Observation shape/range: passed")
    print("Action to Conv_a mapping: passed")
    print("Full episode: passed")
    print("steps:", step_count)
    print("final FE:", swarm.fe_count)
    print("final best:", swarm.gbest_fitness)
    print("episode reward:", episode_reward)


if __name__ == "__main__":
    main()