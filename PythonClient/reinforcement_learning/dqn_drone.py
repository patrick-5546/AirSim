import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback


START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

TEST_MODE = False
MODEL_VERBOSE = 0 if TEST_MODE else 1
ENV_VERBOSE = 1 if TEST_MODE else 0
N_EVAL_EPISODES = 1 if TEST_MODE else 5
EVAL_FREQ = 5 if TEST_MODE else 10_000
TOTAL_TIMESTEPS = 10 if TEST_MODE else 250_000


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 1),
                start_time=START_TIME,
                verbose=ENV_VERBOSE,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    buffer_size=500_000,
    learning_starts=25_000,
    verbose=MODEL_VERBOSE,
    device="cuda",
    tensorboard_log="./drone_out/tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=N_EVAL_EPISODES,
    best_model_save_path="./drone_out/eval/" + START_TIME,
    log_path="./drone_out/eval/" + START_TIME,
    eval_freq=EVAL_FREQ,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks
kwargs["progress_bar"] = True

# Train for a certain number of timesteps
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="dqn_airsim_drone_run_" + START_TIME,
    **kwargs
)

# Save policy weights
model.save("./drone_out/model/dqn_airsim_drone_policy_" + START_TIME)
