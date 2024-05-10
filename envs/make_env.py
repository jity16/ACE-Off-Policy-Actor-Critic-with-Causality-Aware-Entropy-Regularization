import gym
import gymnasium


def build_environment(config):
	print("Building Environment %s with %s reward"%(config.env_name, config.reward_type))
	# * Meta-World (reward_type = sparse/dense)
	if (config.env_name).endswith("goal-observable"):   
		from .metaworld_env import metaworld_env
		env = metaworld_env(config.env_name, config.seed, episode_length=200, reward_type=config.reward_type)
		
	#* MuJoCo-v2
	elif config.env_name.endswith("-v2"):
		env = gym.make(config.env_name)

	# * DMControl
	elif config.env_name.endswith("-v0"):
		import dmc_envs
		env = gym.make(config.env_name)

	# * Gymnasium -> Gym interface
	else:       
		from .gymnasium_env import gymnasium2gymEnv   
		specified_kwargs = config.xml_file if 'xml_file' in config else None
		env = gymnasium2gymEnv(env_name=config.env_name, seed=config.seed, xml_file=specified_kwargs)
	return env

if __name__ == "__main__":
	from config_params.default_config import default_config, dmc_config, metaworld_config
	config = dmc_config
	env = build_environment(config)
	state = env.reset()
	next_obs, reward, done, info = env.step(env.action_space.sample())
	print(next_obs, reward, done, info)
