from collections import deque, defaultdict
from typing import Any, NamedTuple
import numpy as np
import gym
from gym.wrappers import TimeLimit
import warnings
import metaworld
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)

	def reset(self):
		state, info = self.env.reset()
		return state

	def step(self, action):
		next_state, reward, terminated, truncated, info = self.env.step(action)
		done = terminated or truncated
		return next_state, reward, done, info

	def render(self, mode='rgb_array', height=384, width=384, camera_id=0):
		self.env.camera_id=camera_id
		return self.env.render()

class MetaWorldSparseWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
	def reset(self):
		state, info = self.env.reset()
		return state
	
	def step(self, action):
		next_state, reward, terminated, truncated, info = self.env.step(action)
		done = terminated or truncated
		return next_state, reward, done, info

	def render(self, mode='rgb_array', height=384, width=384, camera_id=0):
		self.env.camera_id=camera_id
		return self.env.render()

def metaworld_env(env_name, seed, episode_length, reward_type="dense", render_mode='rgb_array'):
	if reward_type == "dense":
		env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=render_mode,seed=seed)
		env = MetaWorldWrapper(env)
		env = TimeLimit(env, max_episode_steps=episode_length)
	elif reward_type == "sparse":
		env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=render_mode,seed=seed)
		env = MetaWorldSparseWrapper(env)
		env = TimeLimit(env, max_episode_steps=episode_length)
	return env