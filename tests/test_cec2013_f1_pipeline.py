import numpy as np

from learning_ddpg.configs.ccpso_config import DIMENSIONS, PARTICLES
from learning_ddpg.environments.factory import make_cec2013_f1_env
from learning_ddpg.functions.cec2013 import CEC2013Objective, CEC2013_OPTIMA


def main():
    objective = CEC2013Objective(
        dimensions=DIMENSIONS,
        function_id=1,
    )
    optimum = CEC2013_OPTIMA[1]

    optimum_value = objective(objective.shift)
    assert np.isclose(optimum_value, optimum, atol=1e-9)

    batch = np.stack([objective.shift, np.zeros(DIMENSIONS)], axis=0)
    values = objective(batch)
    assert values.shape == (2,)
    assert np.isclose(values[0], optimum, atol=1e-9)
    assert np.all(values >= optimum - 1e-9)

    env = make_cec2013_f1_env()
    observation, info = env.reset(seed=42)
    assert observation.shape == (5,)
    assert info["function_id"] == 1
    assert np.isclose(info["function_optimum"], optimum)
    assert np.isclose(
        info["gap"],
        info["gbest_fitness"] - optimum,
    )
    assert info["fe_count"] == PARTICLES

    while True:
        action = np.array([0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert info["gbest_fitness"] >= optimum - 1e-9
        assert np.isclose(
            info["gap"],
            info["gbest_fitness"] - optimum,
        )
        if terminated or truncated:
            break

    assert info["fe_count"] == env.swarm.max_fe
    env.close()

    print("Original CEC2013 F1 objective: passed")
    print("F1 optimum -1400 and gap calculation: passed")
    print("Shared F1 environment factory: passed")
    print(f"final raw fitness={info['gbest_fitness']:.6e}")
    print(f"final optimality gap={info['gap']:.6e}")


if __name__ == "__main__":
    main()
