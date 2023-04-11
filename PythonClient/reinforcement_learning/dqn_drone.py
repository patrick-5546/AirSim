import argparse
import gym
import setup_path
import time

import airgym
from airgym.envs.drone_env import PathSelection, DistMode

from enum import Enum
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
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
    args = get_args()
    env = get_env(args)
    model = get_model(args, env)
    execute(args, env, model)


def get_args():
    """Argparse configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=RLAlgorithm, type=RLAlgorithm, help="RL algorithm to use")
    parser.add_argument("-p", "--path", choices=PathSelection, type=PathSelection, help="Path to use")
    parser.add_argument("-d", "--dist_mode", choices=DistMode, type=DistMode, help="how distance is computed")
    parser.add_argument("-l", "--load", help="path to the model zip file to load; if not specified start from scratch")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model if set, else train")
    parser.add_argument("-s", "--spoof", action="store_true", help="gps spoofing: make position data unreliable")
    parser.add_argument("-t", "--test", action="store_true",
                        help="test mode: small total timesteps and increase verbosity")
    return parser.parse_args()


def get_env(args):
    # create a DummyVecEnv for main airsim gym env
    env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "airgym:airsim-drone-sample-v0",
                    ip_address="127.0.0.1",
                    step_length=0.25,
                    image_shape=(84, 84, 1),
                    start_time=START_TIME,
                    verbose=1 if args.test else 0,
                    target_path=args.path,
                    dist_mode=args.dist_mode,
                    gps_spoofing=args.spoof,
                )
            )
        ]
    )

    # wrap env as VecTransposeImage to allow SB to handle frame observations
    env = VecTransposeImage(env)

    return env


def get_model(args, env):
    """Create model or load from file."""
    if not args.load:
        # Initialize RL algorithm type and parameters
        common_kwargs = {
            "policy": "CnnPolicy",
            "env": env,
            "verbose": 0 if args.test else 1,
            "device": "cuda",
            "tensorboard_log": f"./{LOG_DIR}/tb_logs/",
        }

        print("initializing", args.algorithm, "model")
        if args.algorithm == RLAlgorithm.DQN:
            model = DQN(
                buffer_size=100_000,
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

        # print the saved hyperparameters
        print("loaded:", "gamma =", model.gamma, "num_timesteps =", model.num_timesteps)

        # as the environment is not serializable, we need to set a new instance of the environment
        model.set_env(env)

    return model


def execute(args, env, model):
    """Train or evaluate model."""
    if not args.evaluate:
        # Create an evaluation callback with the same env
        callbacks = []
        eval_callback = EvalCallback(
            env,
            callback_on_new_best=None,
            n_eval_episodes=1 if args.test else 5,
            best_model_save_path=f"./{LOG_DIR}/eval/{START_TIME}",
            log_path=f"./{LOG_DIR}/eval/{START_TIME}",
            eval_freq=5 if args.test else 10_000,
        )
        callbacks.append(eval_callback)

        kwargs = {
            "callback": callbacks,
            "progress_bar": True,
        }

        # Train for a certain number of timesteps
        model.learn(
            total_timesteps=10 if args.test else 100_000,
            tb_log_name=f"{args.algorithm}_airsim_drone_run_{START_TIME}",
            **kwargs
        )

        # Save policy weights
        model.save(f"./{LOG_DIR}/model/{args.algorithm}_airsim_drone_policy_{START_TIME}")
    else:
        if not args.load:
            raise ValueError("Specify model to evaluate with -l/--load")

        n_eval_episodes = 5 if args.test else 20
        print(f'Evaluating model for {n_eval_episodes} episodes')
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == "__main__":
    main()
