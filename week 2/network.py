#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Neural network architecture for the DQN agent.
A standard CNN that takes stacked frames and outputs
Q-values for each action.
"""


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[5]:


class QNetwork(nn.Module):
    """
    Convolutional Q-Network.

    Input:  (batch, n_frames, 84, 84)
    Output: (batch, n_actions)
    """

    def __init__(self, n_frames, n_actions):
        """
        Define the layers:
          - 3 convolutional layers (with ReLU)
          - 2 fully connected layers
        Args:
            n_frames: int, number of stacked frames (input channels)
            n_actions: int, number of possible actions
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=n_frames,
            out_channels=32,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: torch.Tensor of shape (batch, n_frames, 84, 84)

        Returns:
            torch.Tensor of shape (batch, n_actions)
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# In[ ]:




