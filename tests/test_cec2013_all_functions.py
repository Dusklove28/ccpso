import numpy as np

from learning_ddpg.configs.ccpso_config import DIMENSIONS, FUNCTION_IDS
from learning_ddpg.functions.cec2013 import CEC2013Objective, CEC2013_OPTIMA


def main():
    rng = np.random.default_rng(20260715)
    points = rng.uniform(-100.0, 100.0, size=(4, DIMENSIONS))

    for function_id in FUNCTION_IDS:
        objective = CEC2013Objective(DIMENSIONS, function_id)
        values = objective(points)

        assert values.shape == (len(points),)
        assert np.isfinite(values).all()
        assert objective.optimum == CEC2013_OPTIMA[function_id]

        print(
            f"F{function_id:02d} passed | "
            f"optimum={objective.optimum:.1f} | "
            f"sample_min={np.min(values):.6e}"
        )

    print("CEC2013 F1-F28 batch adapter: passed")


if __name__ == "__main__":
    main()
