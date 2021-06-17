import os
import numpy as np
import datetime
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
                np.array(self.probs), np.array(self.vals),  \
                np.array(self.rewards), np.array(self.dones), \
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, action_std, input_dims, alpha,
               fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, self.n_actions*2)
        )

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=alpha)

        self.checkpoint_file = 'Saves/agent_%s/actor_ep_' % datetime.datetime.now().strftime("%m%d-%H%M")

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        actor_output = self.actor(state)

        # Split output into two vectors for the parameterized policy
        action_means = T.tanh(actor_output[:, :self.n_actions]).squeeze(0)
        stds = F.softplus(actor_output[:, self.n_actions:]).squeeze(0)

        dist = Normal(action_means, stds)
        return dist

    def save_checkpoint(self, e):
        T.save(self.state_dict(), self.checkpoint_file + str(e))

    def load_checkpoint(self, e):
        self.load_state_dict(T.load(self.checkpoint_file + str(e)))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='Saves'):
        super(CriticNetwork, self).__init__()


        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=alpha)

        self.checkpoint_file = 'Saves/agent_%s/critic_ep_' % datetime.datetime.now().strftime("%m%d-%H%M")

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self, e):
        T.save(self.state_dict(), self.checkpoint_file + str(e))

    def load_checkpoint(self, e):
        self.load_state_dict(T.load(self.checkpoint_file + str(e)))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, glambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, action_std=0.1,
                 writer=None, actor_lr=1e-4, critic_lr=2e-3):
        self.w = writer
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.tot_epochs = 0
        self.glambda = glambda

        self.actor = ActorNetwork(n_actions, action_std, input_dims, actor_lr)
        self.critic = CriticNetwork(input_dims, critic_lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, e):
        print("Saving models...")

        if not os.path.exists(self.actor.checkpoint_file):
            os.makedirs(self.actor.checkpoint_file)

        self.actor.save_checkpoint(e)
        self.critic.save_checkpoint(e)

    def load_models(self, path, e):
        print("Loading models...")
        self.actor.checkpoint_file = path + "actor_ep_"
        self.critic.checkpoint_file = path + "critic_ep_"

        self.actor.load_checkpoint(e)
        self.critic.load_checkpoint(e)

    def act(self, obs, test=False):
        state = T.tensor([obs], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)

        #Choose between exploration or evaluation
        if test is False:
            action = dist.sample()
        else:
            action = dist

        probs = T.squeeze(dist.log_prob(action).sum(axis=-1)).detach().cpu().numpy()
        action = action.detach().cpu().numpy()
        value = T.squeeze(value).detach().cpu().item()

        return action, probs, value

    def learn(self):
        for e in range(self.n_epochs):
            #Get batches
            state_arr, actions_arr, old_probs_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                        self.memory.generate_batches()

            values = vals_arr

            #Compute GAE advantages
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                adv = 0
                for k in range(t, len(reward_arr)-1):
                    adv += discount*(reward_arr[k] + self.gamma*values[k+1] *
                                     (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.glambda

                advantage[t] = adv

            self.w.add_scalar('Epoch/Advantage', advantage.mean(), self.tot_epochs)

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)


            b = 0
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(actions_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                self.w.add_scalar("Batch/ Entropy", dist.entropy().mean(), self.tot_epochs + b)

                # Get state value
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                # Get log probs
                new_probs = dist.log_prob(actions).sum(axis=-1)

                # Compute ratio
                prob_ratio = T.exp(new_probs-old_probs)

                self.w.add_scalar("Batch/ Probs ratio", prob_ratio.mean().detach(), self.tot_epochs + b)

                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                        1+self.policy_clip)*advantage[batch]

                # Compute surrogate objective
                actor_loss = -T.min(weighted_probs, clipped_probs).mean()
                self.w.add_scalar('Batch/ Actor Loss', actor_loss.detach(), self.tot_epochs + b)

                returns = advantage[batch] + values[batch]

                # Compute MSE
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                self.w.add_scalar('Batch/ Critic Loss', critic_loss.detach(), self.tot_epochs + b)

                total_loss = actor_loss + 0.5*critic_loss
                self.w.add_scalar('Batch/ Total Loss', critic_loss.detach(), self.tot_epochs + b)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                b += 1
            self.tot_epochs += 1

        self.memory.clear_memory()