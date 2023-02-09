import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback


TEST_MODE = True


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
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
    learning_rate=0.00025,
    verbose=1,
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

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./car_out/eval",
    log_path="./car_out/eval",
    eval_freq=5 if TEST_MODE else 10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks
kwargs["progress_bar"] = True

# Train for a certain number of timesteps
model.learn(
    total_timesteps=10 if TEST_MODE else 5e5,
    tb_log_name="dqn_airsim_car_run_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
    **kwargs
)

# Save policy weights
model.save("./car_out/dqn_airsim_car_policy")
