#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
Double DQN Agent with Prioritized Experience Replay.
"""
import import_ipynb
import numpy as np
import torch
import torch.optim as optim

from network import QNetwork
from replay_buffer import PrioritizedReplayBuffer


class DoubleDQNAgent:
    """
    Encapsulates:
      - online and target Q-networks
      - epsilon-greedy action selection
      - Double DQN learning step
      - target network syncing
    """

    def __init__(self, n_frames, n_actions, config, device="cpu"):
        """
        Args:
            n_frames:  int, number of stacked frames
            n_actions: int, number of possible actions
            config:    Config object with hyperparameters
            device:    str, "cpu" or "cuda"
        """
        self.n_actions = n_actions
        self.config = config
        self.device = device
        self.step_count = 0

        # Networks
        self.online_net = QNetwork(n_frames, n_actions).to(device)
        self.target_net = QNetwork(n_frames, n_actions).to(device)
        self.sync_target()

        # Optimiser
        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=config.LEARNING_RATE
        )

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            beta_start=config.PER_BETA_START,
            beta_end=config.PER_BETA_END,
            beta_steps=config.PER_BETA_STEPS,
            epsilon=config.PER_EPSILON,
        )
    def sync_target(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )


    def get_epsilon(self):
        """
        Linear epsilon decay.
        """
        fraction = min(
            self.step_count / self.config.EPSILON_DECAY_STEPS,
            1.0
        )

        epsilon = (
            self.config.EPSILON_START
            + fraction
            * (self.config.EPSILON_END - self.config.EPSILON_START)
        )

        return epsilon


    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        epsilon = self.get_epsilon()

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        state = torch.FloatTensor(state)\
            .unsqueeze(0)\
            .to(self.device)

        with torch.no_grad():
            q_values = self.online_net(state)

        return q_values.argmax(dim=1).item()


    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.buffer.store(
            state,
            action,
            reward,
            next_state,
            done
        )


    def learn(self):

        if len(self.buffer) < self.config.BATCH_SIZE:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights
        ) = self.buffer.sample(
            self.config.BATCH_SIZE
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        #
        # Current Q-values
        #
        current_q = self.online_net(states)\
            .gather(1, actions.unsqueeze(1))\
            .squeeze(1)

        #
        # Double DQN
        #
        with torch.no_grad():

            # action selection
            next_actions = self.online_net(
                next_states
            ).argmax(dim=1)

            # action evaluation
            next_q = self.target_net(
                next_states
            ).gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            targets = (
                rewards
                + self.config.GAMMA
                * next_q
                * (1 - dones)
            )

        #
        # TD errors
        #
        td_errors = targets - current_q

        #
        # PER weighted loss
        #
        loss = (
            weights
            * td_errors.pow(2)
        ).mean()

        self.optimiser.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            10.0
        )

        self.optimiser.step()

        #
        # Update PER priorities
        #
        new_priorities = (
            td_errors.detach()
            .abs()
            .cpu()
            .numpy()
        )

        self.buffer.update_priorities(
            indices,
            new_priorities
        )

        self.step_count += 1

        #
        # Target network update
        #
        if self.step_count % self.config.TARGET_UPDATE == 0:
            self.sync_target()

        return loss.item()


    def save(self, path):
        """Save model weights to disk."""
        torch.save(
            self.online_net.state_dict(),
            path
        )


    def load(self, path):
        """Load model weights from disk."""
        self.online_net.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.sync_target()


# In[ ]:





# In[ ]:




