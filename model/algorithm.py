import os
import numpy as np
import ipdb
import torch
import copy
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .dormant_utils import cal_dormant_ratio, cal_dormant_grad, neurons_impact, similarity, perturb, dormant_perturb, perturb_factor
from .model import GaussianCausalPolicy, ValueNetwork, QNetwork



class ACE_agent(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.reset = args.reset
        self.reset_interval = args.reset_interval

        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GaussianCausalPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    #* get Q value given a batch of memory
    def get_Q_value(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        qf1, qf2 = self.critic(torch.FloatTensor(state_batch).to(self.device), torch.FloatTensor(action_batch).to(self.device))
        return state_batch, action_batch, torch.min(qf1, qf2)

    def update_parameters(self, memory, causal_weight, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        dormant_metrics = {}
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, causal_weight)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        if updates == 0:
            self.policy_loss = 0*qf_loss
            self.alpha_loss = 0*qf_loss
            self.alpha_tlogs = 0*qf_loss

        if updates % self.target_update_interval == 0:
            pi, log_pi, _ = self.policy.sample(state_batch, causal_weight)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
        
            soft_update(self.critic_target, self.critic, self.tau)
            self.policy_loss = policy_loss
            self.alpha_loss = alpha_loss
            self.alpha_tlogs = alpha_tlogs
        
        if updates % self.reset_interval == 0 and updates > 5000:
            # dormant_metrics, _, _ = cal_dormant_ratio(self.policy, state_batch, type='policy', percentage=0.1)
            dormant_metrics = cal_dormant_grad(self.policy, type='policy', percentage=0.05)

            # dormant_metrics.update(grad_metrics)
            
            if dormant_metrics:
                factor = perturb_factor(dormant_metrics['policy_grad_dormant_ratio'])
            else:
                factor = 1

            causal_diff = np.max(causal_weight) - np.min(causal_weight)
            dormant_metrics["causal_diff"] = causal_diff
            
            if self.reset == 'causal_reset' or self.reset == 'causal_dormant_reset':
                causal_factor = np.exp(-8 * causal_diff)-0.5
                factor = perturb_factor(causal_factor)
                
            dormant_metrics["factor"] =factor

            if factor < 1: 
                if self.reset == 'reset' or self.reset == 'causal_reset':
                    perturb(self.policy, self.policy_optim, factor)
                    perturb(self.critic, self.critic_optim, factor)
                    perturb(self.critic_target, self.critic_optim, factor)
                elif self.reset == 'dormant_reset' or self.reset == 'causal_dormant_reset':
                    self.policy, self.policy_optim = dormant_perturb(self.policy, self.policy_optim, dormant_indices, factor)
                    perturb(self.critic, self.critic_optim, factor)
                    perturb(self.critic_target, self.critic_optim, factor)

        return qf1_loss.item(), qf2_loss.item(), self.policy_loss.item(), self.alpha_loss.item(), self.alpha_tlogs.item(), next_q_value.mean().item(), dormant_metrics

    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    def load_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

    def load_policy_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
            else:
                self.policy.train()

    def load_checkpoint_from_cpu(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()



