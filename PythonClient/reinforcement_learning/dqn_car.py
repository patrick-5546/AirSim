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
N_EVAL_EPISODES = 1 if TEST_MODE else 5
EVAL_FREQ = 5 if TEST_MODE else 10000
TOTAL_TIMESTEPS = 10 if TEST_MODE else 5e5

LOAD_MODEL = False
LOAD_START_TIME = "2023-02-10_15-44-05"


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
                start_time=START_TIME,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

if not LOAD_MODEL:
    # Initialize RL algorithm type and parameters
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=0.00025,
        verbose=MODEL_VERBOSE,
        batch_size=32,
        train_freq=4,
        target_update_interval=10000,
        learning_starts=200000,
        buffer_size=500000,
        max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        device="cuda",
        tensorboard_log="./car_out/tb_logs/",
    )
else:
    print("loading best model with start time", LOAD_START_TIME)
    model = DQN.load("./car_out/eval/" + LOAD_START_TIME + "/best_model")

    # show the save hyperparameters
    print("loaded:", "gamma =", model.gamma, "num_timesteps =", model.num_timesteps)

    # as the environment is not serializable, we need to set a new instance of the environment
    model.set_env(env)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=N_EVAL_EPISODES,
    best_model_save_path="./car_out/eval/" + START_TIME,
    log_path="./car_out/eval/" + START_TIME,
    eval_freq=EVAL_FREQ,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks
kwargs["progress_bar"] = True

# Train for a certain number of timesteps
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="dqn_airsim_car_run_" + START_TIME,
    **kwargs
)

# Save policy weights
model.save("./car_out/model/dqn_airsim_car_policy_" + START_TIME)
