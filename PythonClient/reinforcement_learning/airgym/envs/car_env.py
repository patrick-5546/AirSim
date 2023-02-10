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


LOG_DIR = 'car_out/env_logs'
REWARD_COL_WIDTH = 6
EPISODE_COL_WIDTH = 7
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

        self.episode = 1
        self.step_obs = False
        self.obs = []
        self.action = ''
        self.actions = []
        self.log_path = os.path.join(LOG_DIR, f'env_log_{start_time}.txt')
        self.eval_episode = False
        self.rewards = []

        os.makedirs(LOG_DIR, exist_ok=True)
        for filename in os.listdir(LOG_DIR):
            filepath = os.path.join(LOG_DIR, filename)
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

    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        done_reason = ""
        if dist > THRESH_DIST:
            reward = -3
            done_reason = "too off course"
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed
            done_reason = f"reward dist{{{reward_dist:.2f}}} + speed{{{reward_speed:.2f}}} < -1"

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
                done_reason = "brake == 0 && speed <= 1"
        if self.state["collision"]:
            done = 1
            done_reason = "collision == True"

        return reward, done, done_reason

    def _log(self, reward, done, done_reason, log_obs=False):
        self.actions.append(self.action)
        self.rewards.append(reward)

        if done:
            if log_obs:
                ep_log_dir = os.path.join(LOG_DIR, str(self.episode))
                os.makedirs(ep_log_dir, exist_ok=True)
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

            self.episode += 1

    def step(self, action):
        self._do_action(action)
        self.step_obs = True
        obs = self._get_obs()
        self.step_obs = False
        reward, done, done_reason = self._compute_reward()
        self._log(reward, done, done_reason)
        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        self.obs = []
        self.actions = []
        self.rewards = []
        return self._get_obs()
