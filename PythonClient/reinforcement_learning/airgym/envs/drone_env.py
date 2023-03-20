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


LEN_TIMESTEP = 2

# logging
START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
LOG_DIR = os.path.join('drone_out', 'env_logs')
OBS_DIR = os.path.join(LOG_DIR, 'obs')
EPISODE_COL_WIDTH = 7
REWARD_COL_WIDTH = 7
DONE_REASON_COL_WIDTH = 35


class Path():
    """How to define a path

    1. Select points in path
        1. Check points in path by uncommenting block in `_setup_flight()`
    2. Select start position, (x, y, z, speed), where `speed` is the navigation speed to the starting position
        1. Run dqn_drone.py in test mode so that the position, path, and distance to path is printed
        2. Finetune (x, y, z) such that distance is minimized (<1)
            1. Because of momentum, the greater `speed` is, the further from (x, y, z) you will be
    """
    # neighborhood paths
    NH_0_START_POS = (0, 0, -8.3, 5)
    NH_0 = [
        np.array([x, y, -9])
        for x, y in [
            (0, 0), (128, 0), (128, 127), (0, 127),
            (0, 0), (128, 0), (128, -128), (0, -128),
            (0, 0),
        ]
    ]
    NH_1_START_POS = NH_0_START_POS
    NH_1 = NH_0[:2]
    # landscape mountain paths
    LM_0_START_POS = (0, -24, -15.5, 10)
    LM_0 = [
        np.array([x, y, z])
        for x, y, z in [
            (-0.55265, -31.9786, -19.0225),
            (48.59735, -63.3286, -60.07256),
            (193.5974, -55.0786, -46.32256),
            (369.2474, 35.32137, -62.5725),
            (541.3474, 143.6714, -32.07256),
        ]
    ]

    def __init__(self, name):
        self.name = name

        if name == 'NH_0':
            self.start_pos = Path.NH_0_START_POS
            self.path = Path.NH_0
        elif name == 'NH_1':
            self.start_pos = Path.NH_1_START_POS
            self.path = Path.NH_1
        elif name == 'LM_0':
            self.start_pos = Path.LM_0_START_POS
            self.path = Path.LM_0
        else:
            raise NameError(f'Path {name} not found')

        assert len(self.path) >= 2, f'Path {name} length {len(self.path)} < 2; path={self.path_str()}'

    def path_str(self):
        return str([f"({x:.0f}, {y:.0f}, {z:.0f})" for x, y, z in self.path])


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, target_path, start_time, verbose):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.target_path = Path(target_path)
        print(f'Setting target path to {target_path}: {self.target_path.path_str()}')

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

        self.path_seg = 1

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

        self.drone.moveToPositionAsync(*self.target_path.start_pos).join()
        self.drone.moveByVelocityAsync(0, 0, 0, LEN_TIMESTEP).join()

        '''
        # check points in path
        for pt in self.target_path.path:
            print(f'Navigating to {pt}')
            self.drone.moveToPositionAsync(*pt, self.target_path.start_pos[3]).join()
            self.drone.moveByVelocityAsync(0, 0, 0, LEN_TIMESTEP).join()
        '''

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
            LEN_TIMESTEP,
        ).join()

    def _compute_reward(self):
        THRESH_DIST = 4
        # graph of distance and speed reward functions: https://www.desmos.com/calculator/j4phheyncf
        # reward function constants
        # x intercept of distance function should be approximately half THRESH_DIST
        DIST_DECAY = 0.3
        SPEED_DECAY = 0.7

        if self.state["collision"]:
            reward = -100
            done = 1
            done_reason = 'collision'
        else:
            dist, reached_destination = self._get_dist(THRESH_DIST)
            if reached_destination:
                reward = 1000
                done = 1
                done_reason = 'reached destination'
            elif dist > THRESH_DIST:
                reward = -10
                done = 1
                done_reason = f'dist{{{dist:.2f}}}>THRESH_DIST{{{THRESH_DIST:.2f}}}'
            else:
                reward_dist = math.exp(-DIST_DECAY * dist) - 0.5
                speed = np.linalg.norm([
                    self.state["velocity"].x_val,
                    self.state["velocity"].y_val,
                    self.state["velocity"].z_val,
                    ])
                reward_speed = -math.exp(-SPEED_DECAY * speed) + 0.5

                reward = reward_dist + reward_speed
                done = 0
                done_reason = f'r_dist{{{reward_dist:.2f}}}+r_speed{{{reward_speed:.2f}}}<=-10'
                if self.verbose:
                    print(f'{dist=:.2f}', f'{reward_dist=:.2f}', sep=' ', end=' ')
                    print(f'{speed=:.2f}', f'{reward_speed=:.2f}', sep=' ', end=' ')
                    print(f'{reward=:.2f}')

        return reward, done, done_reason

    def _get_dist(self, thresh_dist):
        # position
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        i = self.path_seg - 1
        # distance to current path segment
        dist = pnt2line(quad_pt, self.target_path.path[i], self.target_path.path[i + 1])[0]
        next_path_seg = self.path_seg + 1
        j = next_path_seg - 1

        if next_path_seg == len(self.target_path.path):
            # reached destination if close enough
            next_dist = np.linalg.norm(self.target_path.path[j] - quad_pt)
            if next_dist <= thresh_dist:
                print('reached destination')

                return next_dist, True
        else:
            # advance to next line segment if close enough
            next_dist = pnt2line(quad_pt, self.target_path.path[j], self.target_path.path[j + 1])[0]
            if next_dist <= thresh_dist:
                print('advancing to next line segment')

                self.path_seg = next_path_seg
                dist = next_dist

        if self.verbose:
            print(f'path_seg={self.path_seg}/{len(self.target_path.path)}', end=' ')

            def format_float_list(list_):
                return '[{:.2f},{:.2f},{:.2f}]'.format(*list_)

            def format_int_list(list_):
                return '[{},{},{}]'.format(*list_)

            dist_pt = self.target_path.path[self.path_seg - 1]
            print(f'quad_pt={format_float_list(quad_pt)}', f'dist_pt={format_int_list(dist_pt)}', sep=' ', end=' ')

            if self.path_seg != next_path_seg:
                next_pt = self.target_path.path[j]
                print(f'next_pt={format_int_list(next_pt)}', f'{next_dist=:.2f}', sep=' ', end = ' ')

        return dist, False

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

        self.path_seg = 1

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
