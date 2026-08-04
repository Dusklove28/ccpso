import numpy as np

from learning_ddpg.environments.factory import make_cec2013_env


REPRESENTATIVE_FUNCTION_IDS = (1, 14, 15, 21, 28)


def main():
    for function_id in REPRESENTATIVE_FUNCTION_IDS:
        env = make_cec2013_env(function_id)
        try:
            state, reset_info = env.reset(seed=42)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_state, reward, terminated, truncated, step_info = env.step(action)

            assert state.shape == env.observation_space.shape
            assert next_state.shape == env.observation_space.shape
            assert env.action_space.contains(action)
            assert np.isfinite(state).all()
            assert np.isfinite(next_state).all()
            assert np.isfinite(reward)
            assert reset_info["function_id"] == function_id
            assert step_info["function_id"] == function_id
            assert np.isclose(
                step_info["gap"],
                step_info["gbest_fitness"] - step_info["function_optimum"],
            )
            assert not terminated
            assert not truncated

            print(
                f"F{function_id:02d} shared env passed | "
                f"state={state.shape} | "
                f"reward={reward:.6f} | "
                f"gap={step_info['gap']:.6e}"
            )
        finally:
            env.close()

    print("Representative CEC2013 shared environments: passed")


if __name__ == "__main__":
    main()
