"""Configuration for the minimal PyTorch CCPSO/DDPG experiment."""

# Match the original RL_Tensorflow_5step seed policy. Training, model
# selection, and final testing use fixed, disjoint seed blocks.
BASE_SEED = 20_260_629
TRAIN_SEED_OFFSET = 0
VALIDATION_SEED_OFFSETS = (50_000, 60_000, 70_000)
TEST_SEED_OFFSET = 100_000

TRAIN_RUNS = 1
VALIDATION_RUNS_PER_OFFSET = 10
TEST_RUNS = 10

TRAIN_SEEDS = tuple(
    BASE_SEED + TRAIN_SEED_OFFSET + run_index
    for run_index in range(TRAIN_RUNS)
)
VALIDATION_SEEDS = tuple(
    BASE_SEED + seed_offset + run_index
    for seed_offset in VALIDATION_SEED_OFFSETS
    for run_index in range(VALIDATION_RUNS_PER_OFFSET)
)
TEST_SEEDS = tuple(
    BASE_SEED + TEST_SEED_OFFSET + run_index
    for run_index in range(TEST_RUNS)
)

DEVICE = "cpu"
TRAIN_EPISODES = 100
CHECKPOINT_INTERVAL = 10
BATCH_SIZE = 64
LEARNING_STARTS = 64
REPLAY_CAPACITY = 10_000

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005
NOISE_STD = 0.1

PARTICLES = 20
DIMENSIONS = 10
FUNCTION_IDS = tuple(range(1, 29))
LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
MAX_FE = 1000
CONV_MIN = 0.0
CONV_MAX = 1.5

STAGE_FE = 100
STAGE_ACTION_MODE = "c_target_hold"
STAGE_SMOOTHING_ALPHA = 0.5
STAGE_MAX_DELTA_C = 0.2
CONTROL_LEVEL = "stage"

# Stage actors must not be mixed with earlier one-swarm-step actors.
CHECKPOINT_EXPERIMENT_NAME = "cec2013_f1_f28_stage_d10_seed20260629"
