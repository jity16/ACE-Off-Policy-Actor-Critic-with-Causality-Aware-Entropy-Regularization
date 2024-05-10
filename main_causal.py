import os
import sys
import torch
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from model.algorithm import ACE_agent
from utilis.Replaybuffer import ReplayMemory
from utilis.causal_weight import get_sa2r_weight
import datetime
import itertools
from copy import copy
import shutil
import wandb
import csv
import yaml
import ipdb
import gym
import random
from envs.make_env import build_environment

def train_loop(config, msg = "default"):
    sys.path.append(os.path.join(os.path.dirname(__file__), 'envs')) 
    env = build_environment(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Agent
    agent = ACE_agent(env.observation_space.shape[0], env.action_space, config)


    result_path = './results/{}/{}/{}_{}_{}_{}_{}'.format(config.env_name, msg, 
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                      config.policy, config.seed, 
                                                      "autotune" if config.automatic_entropy_tuning else "",
                                                      config.msg)
    

    checkpoint_path = result_path + '/' + 'checkpoint'
    
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))
    
    #* Logging Causal weight
    causal_weight_csv_file = os.path.join(result_path, "causal_weight.csv")
    with open(causal_weight_csv_file, mode='w', newline='') as csv_file:
        fieldnames = ['Time Step', 'Causal Weights']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()



    # saving code
    current_path = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(current_path)
    files_to_save = ['main.py', 'model','utilis']
    ignore_files = [x for x in files if x not in files_to_save]
    shutil.copytree('.', result_path + '/code', ignore=shutil.ignore_patterns(*ignore_files))
    
    memory = ReplayMemory(config.replay_size, config.seed)
    local_buffer = ReplayMemory(config.causal_sample_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    best_success = 0.0
    causal_computing_time = 0.0
    causal_weight = np.ones(env.action_space.shape[0], dtype=np.float32)
    W_est = []
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        state = env.reset()
        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > config.batch_size:
                for i in range(config.updates_per_step):
                    #* Update parameters of causal weight
                    if (total_numsteps % config.causal_sample_interval == 0) and (len(local_buffer)>=config.causal_sample_size):
                        causal_weight, causal_computing_time = get_sa2r_weight(env, local_buffer, agent, sample_size=config.causal_sample_size, causal_method='DirectLiNGAM')
                        print("Current Causal Weight is: ",causal_weight)
                        wandb.log(
                            data={
                                'Causal/Computing Time': causal_computing_time,
                            },
                            step = total_numsteps
                        )
                        with open(causal_weight_csv_file, mode='a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow([total_numsteps, ', '.join(map(str, causal_weight))])

                    dormant_metrics = {}
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, q_sac, dormant_metrics = agent.update_parameters(memory, causal_weight,config.batch_size, updates)

                    wandb.log(dormant_metrics)
                    wandb.log(
                        data = {
                            'loss/q_critic_1': critic_1_loss,
                            'loss/q_critic_2': critic_2_loss,
                            'loss/policy_loss': policy_loss,
                            'loss/entropy_loss': ent_loss, 
                        },
                        step = total_numsteps
                    )
                    
                    updates += 1
            next_state, reward, done, info = env.step(action) # Step
            total_numsteps += 1
            episode_steps += 1
            episode_reward += reward

            #* Ignore the "done" signal if it comes from hitting the time horizon.
            if '_max_episode_steps' in dir(env):  
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            elif 'max_path_length' in dir(env):
                mask = 1 if episode_steps == env.max_path_length else float(not done)
            else: 
                mask = 1 if episode_steps == 1000 else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            local_buffer.push(state, action, reward, next_state, mask) # Append transition to local_buffer
            state = next_state

        if total_numsteps > config.num_steps:
            break


        wandb.log(
            data={
                'reward/train_reward': episode_reward
            },
            step = total_numsteps
        )
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # test agent
        if i_episode % config.eval_interval == 0 and config.eval is True:
            eval_reward_list = []
            for _  in range(config.eval_episodes):
                state = env.reset()
                episode_reward = []
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    episode_reward.append(reward)
                eval_reward_list.append(sum(episode_reward))

            avg_reward = np.average(eval_reward_list)
            
            if config.save_checkpoint:
                if avg_reward >= best_reward:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')



            wandb.log(
                data = {
                    'reward/test_avg_reward': avg_reward,
                },
                step = total_numsteps
            )
          
            print("----------------------------------------")
            print("Env: {}, Algo:{},  Test Episodes: {}, Avg. Reward: {}".format(config.env_name, config.algo, config.eval_episodes, round(avg_reward, 2)))
            print("----------------------------------------")
    env.close() 


def main():
    arg = ARGConfig()
    arg.add_arg("env_name", "coffee-button-v2-goal-observable", "Environment name")
    arg.add_arg("reward_type", "dense", "sparse or dense")
    arg.add_arg("device", "0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("algo", "ACE", "choose algorithm)")
    arg.add_arg("start_steps", 10000, "Number of start steps")
    arg.add_arg("num_steps", 1000001, "total number of steps")
    arg.add_arg("save_checkpoint", False, "save checkpoint or not")
    arg.add_arg("replay_size", 1000000, "size of replay buffer")
    arg.add_arg("causal_sample_interval", 10000, "sample_size for causal computing")
    arg.add_arg("causal_sample_size", 10000, "sample_size for causal computing")
    arg.add_arg("causal_model","DirectLiNGAM", "causal model type") 
    arg.add_arg("reset", "reset", "Types of reset")
    arg.add_arg("reset_interval", 100000, "Reset interval")
    arg.add_arg("des", "", "short description for the experiment")
    arg.parser()

    config = default_config
    config.update(arg)
    algorithm = config.algo
    config["seed"] = np.random.randint(1000)
    

    experiment_name = "{}-{}-{}-{}-s{}-{}".format(
        config['reward_type'],
        algorithm, 
        config['env_name'], 
        str(config["seed"]), 
        config["causal_sample_interval"],
        config["reset"]
    )
    
    run_id = "{}-{}_{}_{}_{}{}-{}_{}".format(
        config['reward_type'],
        algorithm, 
        config['env_name'],
        str(config["seed"]), 
        config["causal_model"],
        config["causal_sample_interval"],
        config["reset"],
        datetime.datetime.now().strftime("%Y-%m-%d_%H")
    )


    run = wandb.init(
        project = config["project_name"],
        config = {
            "env_name": config['env_name'],
            "algorithm" : algorithm,
            "seed": config["seed"],
            "reset": config["reset"],
            "causal_sample_interval": config["causal_sample_interval"],
            "causal_sample_size": config["causal_sample_size"],
            "num_steps": config["num_steps"]
        },
        name = experiment_name,
        id = run_id,
        save_code = False
    )


    print(f">>>> Training {algorithm} on {config.env_name} environment, on {config.device}")
    train_loop(config, msg=algorithm)
    wandb.finish()


if __name__ == "__main__":
    main()
