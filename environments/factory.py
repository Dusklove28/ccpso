from learning_ddpg.configs.ccpso_config import (
    CONV_MAX,
    CONV_MIN,
    DIMENSIONS,
    FUNCTION_IDS,
    LOWER_BOUND,
    MAX_FE,
    PARTICLES,
    UPPER_BOUND,
    STAGE_FE,
    STAGE_ACTION_MODE,
    STAGE_SMOOTHING_ALPHA,
    STAGE_MAX_DELTA_C
)
from learning_ddpg.environments.ccpso_env import CCPSOEnv
from learning_ddpg.functions.cec2013 import CEC2013Objective
from learning_ddpg.swarm.ccpso import CCPSOSwarm


def make_cec2013_env(function_id):
    function_id = int(function_id)
    if function_id not in FUNCTION_IDS:
        raise ValueError(
            f"F{function_id} is not enabled; configured functions: {FUNCTION_IDS}"
        )

    objective = CEC2013Objective(
        dimensions=DIMENSIONS,
        function_id=function_id,
    )
    swarm = CCPSOSwarm(
        particles=PARTICLES,
        dimensions=DIMENSIONS,
        fun=objective,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        max_fe=MAX_FE,
    )

    return CCPSOEnv(
        swarm=swarm,
        conv_min=CONV_MIN,
        conv_max=CONV_MAX,
        optimum=objective.optimum,
        function_id=function_id,
        stage_fe=STAGE_FE,
        stage_action_mode=STAGE_ACTION_MODE,
        stage_smoothing_alpha=STAGE_SMOOTHING_ALPHA,
        stage_max_delta_c=STAGE_MAX_DELTA_C,
    )


def make_cec2013_f1_env():
    return make_cec2013_env(1)


def make_small_ccpso_env():
    """Backward-compatible name for the configured small F1 environment."""
    return make_cec2013_f1_env()
