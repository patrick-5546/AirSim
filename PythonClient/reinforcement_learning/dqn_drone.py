import argparse
import gym
import setup_path
import time

import airgym

from enum import Enum
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
LOG_DIR = "drone_out"


class RLAlgorithm(Enum):
    DQN = "dqn"
    PPO = "ppo"

    def __str__(self):
        return self.value


def main():
    # argparse configuration
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--load", default=argparse.SUPPRESS,
                        help="path to the model zip file to load; if not specified start from scratch")
    parser.add_argument("-a", "--algorithm", choices=RLAlgorithm, type=RLAlgorithm, default=RLAlgorithm.DQN,
                        help="RL algorithm to use")
    parser.add_argument("-t", "--test", action="store_true",
                        help="test mode: small total timesteps and increase verbosity")
    args = parser.parse_args()

    # parameters dependent on test mode
    model_verbose = 0 if args.test else 1
    env_verbose = 1 if args.test else 0
    n_eval_episodes = 1 if args.test else 5
    eval_freq = 5 if args.test else 10_000
    total_timesteps = 10 if args.test else 250_000

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
                    verbose=env_verbose,
                )
            )
        ]
    )

    # Wrap env as VecTransposeImage to allow SB to handle frame observations
    env = VecTransposeImage(env)

    if not args.load:
        # Initialize RL algorithm type and parameters
        common_kwargs = {
            "policy": "CnnPolicy",
            "env": env,
            "verbose": model_verbose,
            "device": "cuda",
            "tensorboard_log": f"./{LOG_DIR}/tb_logs/",
        }

        print("initializing", args.algorithm, "model")
        if args.algorithm == RLAlgorithm.DQN:
            model = DQN(
                buffer_size=200_000,
                learning_starts=10_000,
                **common_kwargs
            )
        elif args.algorithm == RLAlgorithm.PPO:
            model = PPO(
                **common_kwargs
            )
    else:
        print("loading", args.algorithm, "model from", args.load)
        if args.algorithm == RLAlgorithm.DQN:
            model = DQN.load(args.load)
        elif args.algorithm == RLAlgorithm.PPO:
            model = PPO.load(args.load)

        # start learning right away
        model.learning_starts = 0

        # print the saved hyperparameters
        print("loaded:", "gamma =", model.gamma, "num_timesteps =", model.num_timesteps)

        # as the environment is not serializable, we need to set a new instance of the environment
        model.set_env(env)

    # Create an evaluation callback with the same env
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=f"./{LOG_DIR}/eval/{START_TIME}",
        log_path=f"./{LOG_DIR}/eval/{START_TIME}",
        eval_freq=eval_freq,
    )
    callbacks.append(eval_callback)

    kwargs = {
        "callback": callbacks,
        "progress_bar": True,
    }

    # Train for a certain number of timesteps
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"dqn_airsim_drone_run_{START_TIME}",
        **kwargs
    )

    # Save policy weights
    model.save(f"./{LOG_DIR}/model/dqn_airsim_drone_policy_{START_TIME}")


if __name__ == "__main__":
    main()
