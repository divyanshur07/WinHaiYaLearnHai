#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Prioritized Experience Replay buffer using a Sum Tree.
"""


# In[2]:


import numpy as np


# In[1]:


import numpy as np

class SumTree:
    """
    Binary tree where each leaf stores a priority and
    internal nodes store the sum of their children.
    Allows O(log n) sampling proportional to priority.
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: int, max number of transitions
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_index = 0
        self.size = 0

    def total(self):
        """Return the total sum of all priorities."""
        return self.tree[0]

    def add(self, priority, data):
        """
        Store a transition with the given priority.

        Args:
            priority: float
            data: tuple (state, action, reward, next_state, done)
        """
        # Index of leaf node
        tree_index = self.write_index + self.capacity - 1

        # Store transition
        self.data[self.write_index] = data

        # Update tree
        self.update(tree_index, priority)

        # Move circular buffer pointer
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, priority):
        """
        Update the priority of a leaf node and propagate.

        Args:
            tree_index: int, index in the tree array
            priority: float, new priority
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate change up to root
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        """
        Retrieve a leaf by sampling a value in [0, total()).

        Args:
            value: float

        Returns:
            (tree_index, priority, data)
        """
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            # Reached leaf node
            if left >= len(self.tree):
                leaf = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_index = leaf - (self.capacity - 1)

        return (
            leaf,
            self.tree[leaf],
            self.data[data_index]
        )


# In[2]:


import numpy as np

class PrioritizedReplayBuffer:
    """
    Replay buffer that samples transitions proportional
    to their TD-error priority.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=200_000, epsilon=1e-6):
        """
        Args:
            capacity: int
            alpha: float, how much prioritisation to use (0 = uniform)
            beta_start: float, initial importance-sampling exponent
            beta_end: float, final beta value
            beta_steps: int, steps over which beta is annealed
            epsilon: float, small constant to avoid zero priorities
        """
        self.tree = SumTree(capacity)

        self.alpha = alpha

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps

        self.epsilon = epsilon

        self.max_priority = 1.0
        self.step_count = 0

    def _get_beta(self):
        """Linearly anneal beta from beta_start to beta_end."""

        fraction = min(1.0, self.step_count / self.beta_steps)

        return (
            self.beta_start +
            fraction * (self.beta_end - self.beta_start)
        )

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition with max priority.
        """

        transition = (
            state,
            action,
            reward,
            next_state,
            done
        )

        priority = self.max_priority ** self.alpha

        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """
        Sample a batch proportional to priorities.

        Returns:
            (states, actions, rewards, next_states,
             dones, indices, is_weights)
        """

        beta = self._get_beta()
        self.step_count += 1

        total_priority = self.tree.total()

        segment = total_priority / batch_size

        batch = []
        indices = []
        priorities = []

        for i in range(batch_size):

            a = segment * i
            b = segment * (i + 1)

            value = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(value)

            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        priorities = np.array(priorities)

        sampling_probs = priorities / total_priority

        N = len(self)

        is_weights = (N * sampling_probs) ** (-beta)

        # Normalize for stability
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            is_weights.astype(np.float32)
        )

    def update_priorities(self, indices, td_errors):
        """
        Update priorities after learning.

        Args:
            indices: list of tree indices
            td_errors: np.ndarray of |TD error| values
        """

        td_errors = np.abs(td_errors)

        for idx, error in zip(indices, td_errors):

            priority = (error + self.epsilon) ** self.alpha

            self.tree.update(idx, priority)

            self.max_priority = max(
                self.max_priority,
                priority
            )

    def __len__(self):
        return self.tree.size


# In[ ]:




