import setup_path
import airsim
import numpy as np
import math
import os
import shutil
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


LOG_DIR = os.path.join('drone_out', 'env_logs')
OBS_DIR = os.path.join(LOG_DIR, 'obs')
EPISODE_COL_WIDTH = 7
REWARD_COL_WIDTH = 7
DONE_REASON_COL_WIDTH = 35


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, start_time, verbose):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        # logging
        self.verbose = verbose
        self.step_obs = False
        self.action = ''
        self.actions = []
        self.rewards = []
        self.episode = 0
        self.obs = []
        self.log_path = os.path.join(LOG_DIR, f'env_log_{start_time}.txt')
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

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, -1, -10, 10).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

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
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    def _compute_reward(self):
        thresh_dist = 7
        beta = 1

        z = -10
        pts = [
            np.array([x, y, z])
            for x, y in [
                (0, -1), (128, -1), (128, 127), (0, 127),
                (0, -1), (128, -1), (128, -128), (0, -128),
                (0, -1),
            ]
        ]

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            reward = -100
            done_reason = 'collision'
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    # np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    # / np.linalg.norm(pts[i] - pts[i + 1]),
                    pnt2line(quad_pt, pts[i], pts[i + 1])[0],
                )
            if self.verbose:
                dists = [(pts[i],
                          #   np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                          #   / np.linalg.norm(pts[i] - pts[i + 1]))
                          pnt2line(quad_pt, pts[i], pts[i + 1])[0])
                         for i in range(len(pts)-1)
                         ]
                dist_pt, _ = min(dists, key=lambda x: x[1])

                def format_float_list(list_):
                    return '[{:.2f},{:.2f},{:.2f}]'.format(*list_)

                def format_int_list(list_):
                    return '[{},{},{}]'.format(*list_)

                print(f'quad_pt={format_float_list(quad_pt)}', f'dist_pt={format_int_list(dist_pt)}', sep=' ', end=' ')

            if dist > thresh_dist:
                reward = -10
                done_reason = f'dist{{{dist:.2f}}}>thresh_dist{{{thresh_dist:.2f}}}'
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                speed = np.linalg.norm([
                    self.state["velocity"].x_val,
                    self.state["velocity"].y_val,
                    self.state["velocity"].z_val,
                    ])
                reward_speed = speed - 0.5
                reward = reward_dist + reward_speed
                done_reason = f'r_dist{{{reward_dist:.2f}}}+r_speed{{{reward_speed:.2f}}}<=-10'
                if self.verbose:
                    print(f'{dist=:.2f}', f'{reward_dist=:.2f}', sep=' ', end=' ')
                    print(f'{speed=:.2f}', f'{reward_speed=:.2f}', sep=' ', end=' ')
                    print(f'{reward=:.2f}')

        done = 0
        if reward <= -10:
            done = 1

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
            actions = ''.join(f'{a : <3}' for a in self.actions).strip()
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

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()

        # logging
        self.obs = []
        self.actions = []
        self.rewards = []

        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
            self.action = '+x'
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
            self.action = '+y'
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
            self.action = '+z'
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
            self.action = '-x'
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
            self.action = '-y'
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
            self.action = '-z'
        else:
            quad_offset = (0, 0, 0)
            self.action = '+0'

        return quad_offset

# From https://stackoverflow.com/a/51240898
def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest 
# distance from pnt to the line and the coordinates of the 
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line. 
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)