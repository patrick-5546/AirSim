import setup_path
import airsim
import numpy as np
import math
import os
import shutil
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


LOG_DIR = os.path.join('car_out', 'env_logs')
OBS_DIR = os.path.join(LOG_DIR, 'obs')
EPISODE_COL_WIDTH = 7
REWARD_COL_WIDTH = 6
DONE_REASON_COL_WIDTH = 40


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape, start_time):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.episode = 0
        self.step_obs = False
        self.obs = []
        self.action = ''
        self.actions = []
        self.log_path = os.path.join(LOG_DIR, f'env_log_{start_time}.txt')
        self.eval_episode = False
        self.rewards = []
        self.step_cnt = 0

        os.makedirs(OBS_DIR, exist_ok=True)
        for filename in os.listdir(OBS_DIR):
            filepath = os.path.join(OBS_DIR, filename)
            if os.path.isdir(filepath):
                print('delete episode directory', filepath)
                shutil.rmtree(filepath)
        header = ' | '.join([f'{"Episode" : <{EPISODE_COL_WIDTH}}',
                             f'{"Reward" : <{REWARD_COL_WIDTH}}',
                             f'{"Done Reason" : <{DONE_REASON_COL_WIDTH}}',
                             "Actions"])
        with open(self.log_path, 'w') as f:
            f.write(f'{header}\n')

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.action = 'stop'
        elif action == 1:
            self.car_controls.steering = 0
            self.action = 'north'
        elif action == 2:
            self.car_controls.steering = 0.5
            self.action = 'east'
        elif action == 3:
            self.car_controls.steering = -0.5
            self.action = 'west'
        elif action == 4:
            self.car_controls.steering = 0.25
            self.action = 'northeast'
        else:
            self.car_controls.steering = -0.25
            self.action = 'northwest'

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image  # , ImageEnhance

        image = Image.fromarray(img2d)
        if self.step_obs:
            img = image.convert("L")
            # filter = ImageEnhance.Brightness(img)
            # img = filter.enhance(1.2)
            self.obs.append(img)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self, verbose=False,
                        func_speed=False, done_speed=False,
                        func_length=False):
        # initial done conditions
        if self.state["collision"]:
            reward = 0
            done = True
            done_reason = "collision occurred"
            return reward, done, done_reason

        # reward components
        # graph of reward functions: https://www.desmos.com/calculator/rnget9xoga
        reward_dist, done, done_reason = self._compute_reward_dist(verbose)
        if done:
            return reward_dist, done, done_reason

        reward_speed, done, done_reason = self._compute_reward_speed(verbose, func_speed, done_speed)
        if done:
            return reward_speed, done, done_reason

        # no need to make reward a function of length if reward >= 0
        reward_length, done, done_reason = self._compute_reward_length(verbose, func_length)
        if done:
            return reward_length, done, done_reason

        # bound reward to [0, 1]
        # assumes each reward component is in [0, 1]
        num_funcs = 1
        if func_speed:
            num_funcs += 1
        if func_speed:
            num_funcs += 1
        reward = 1 / num_funcs * (reward_dist + reward_speed + reward_length)
        done = False
        done_reason = ""
        print(f'unbounded reward components:\t{reward_dist = :.2f}\t{reward_speed = :.2f}\t{reward_length = :.2f}')
        return reward, done, done_reason

    def _compute_reward_dist(self, verbose):
        DIST_DECAY = 1
        DIST_THRESH = 4
        PATH = [
            np.array([x, y])
            for x, y in [
                (0, -1), (128, -1), (128, 127), (0, 127),
                (0, -1), (128, -1), (128, -128), (0, -128),
                (0, -1),
            ]
        ]
        PATH -= PATH[0]

        car_pt = self.state["pose"].position.to_numpy_array()[:2]
        if verbose:
            print(f'position = ({car_pt[0]:.2f}, {car_pt[1]:.2f})', end='\t')

        dist = 10000000
        for i in range(0, len(PATH) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - PATH[i]), (car_pt - PATH[i + 1]))
                )
                / np.linalg.norm(PATH[i] - PATH[i + 1]),
            )
        if verbose:
            print(f'{dist = :.2f}', end='\t')

        if dist > DIST_THRESH:
            reward = 0
            done = True
            done_reason = "too off course"
        else:
            reward_dist = math.exp(-DIST_DECAY * dist)
            if verbose:
                print(f'{reward_dist = :.2f}')
            reward, done, done_reason = reward_dist, False, ""
        return reward, done, done_reason

    def _compute_reward_speed(self, verbose, func_speed, done_speed):
        SPEED_MIN = -1
        SPEED_DESIRED = 10
        SPEED_MAX = 25
        SPEED_A = 0.01
        SPEED_C = 1

        speed = self.car_state.speed
        if verbose:
            print(f'{speed = :.2f}', end='\t')

        if func_speed and done_speed and (SPEED_MIN > speed or speed > SPEED_MAX):
            reward = 0
            done = True
            done_reason = "too fast"
        else:
            reward_speed = -SPEED_A * (self.car_state.speed - SPEED_DESIRED) ** 2 + SPEED_C if func_speed else 0
            if verbose:
                print(f'{reward_speed = :.2f}\t{func_speed = }')
            reward, done, done_reason = reward_speed, False, ""
        return reward, done, done_reason

    def _compute_reward_length(self, verbose, func_length):
        LENGTH_DECAY = 0.1

        length = self.step_cnt
        if verbose:
            print(f'{length = }', end='\t')

        reward_length = -math.exp(-LENGTH_DECAY * length) if func_length else 0
        if verbose:
            print(f'{reward_length = :.2f}\t{func_length = }')
        reward, done, done_reason = reward_length, False, ""
        return reward, done, done_reason

    def _log(self, reward, done, done_reason, log_obs=False):
        self.actions.append(self.action)
        self.rewards.append(reward)

        if done:
            self.episode += 1
            print(f'{done_reason=}')

            if log_obs:
                ep_log_dir = os.path.join(OBS_DIR, str(self.episode))
                os.mkdir(ep_log_dir)
                for i, img in enumerate(self.obs):
                    img_name = os.path.join(ep_log_dir, f'{i}.jpg')
                    img.save(img_name)

            episode_reward = np.sum(self.rewards)
            actions = ''.join(f'{a : <10}' for a in self.actions).strip()
            row = ' | '.join([f'{self.episode : <{EPISODE_COL_WIDTH}}',
                              f'{episode_reward : <{REWARD_COL_WIDTH}.2f}',
                              f'{done_reason : <{DONE_REASON_COL_WIDTH}}',
                              actions])
            with open(self.log_path, 'a') as f:
                f.write(f'{row}\n')

    def step(self, action):
        self._do_action(action)
        self.step_obs = True
        obs = self._get_obs()
        self.step_obs = False
        reward, done, done_reason = self._compute_reward()
        self._log(reward, done, done_reason)
        self.step_cnt += 1
        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        self.obs = []
        self.actions = []
        self.rewards = []
        self.step_cnt = 0
        return self._get_obs()
