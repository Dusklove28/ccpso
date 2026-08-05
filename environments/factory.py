from environments.ccpso_env import CCPSOEnv
from problems.spec import ProblemSpec
from swarm.ccpso import CCPSOSwarm


def make_ccpso_env(
        problem: ProblemSpec,
        *,
        particles: int,
        max_fe: int,
        seed: int | None = None,
        c_min: float = 0.0,
        c_max: float = 1.5,
        recent_window: int = 5,
        stagnation_horizon: int = 10,
):
    if not isinstance(problem, ProblemSpec):
        raise TypeError(
            "problem must be an instance of ProblemSpec"
        )

    swarm = CCPSOSwarm(
        particles=particles,
        dimensions=problem.dimensions,
        fun=problem.evaluate,
        lower_bound=problem.lower_bound,
        upper_bound=problem.upper_bound,
        max_fe=max_fe,
        seed=seed,
    )
    env = CCPSOEnv(
        swarm=swarm,
        c_min=c_min,
        c_max=c_max,
        optimum=problem.optimum,
        function_id=problem.problem_id,
        recent_window=recent_window,
        stagnation_horizon=stagnation_horizon,
    )
    env.problem = problem
    return env
