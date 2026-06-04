#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Centralised hyperparameter configuration.
All magic numbers live here so every other module
can simply `from config import Config`.
"""


class Config:
    # Environment
    ENV_NAME        = "ALE/Boxing-v5"
    FRAME_STACK     = 4
    FRAME_SIZE      = 84

    # Network
    LEARNING_RATE   = 1e-4
    BATCH_SIZE      = 32
    GAMMA           = 0.99

    # Epsilon schedule
    EPSILON_START     = 1.0
    EPSILON_END        = 0.01
    EPSILON_DECAY_STEPS = 100_000

    # Replay buffer
    BUFFER_SIZE     = 100_000
    PER_ALPHA       = 0.6
    PER_BETA_START  = 0.4
    PER_BETA_END    = 1.0
    PER_BETA_STEPS  = 200_000
    PER_EPSILON     = 1e-6

    # Training
    TARGET_UPDATE   = 1_000
    TRAIN_START     = 10_000
    TOTAL_STEPS     = 500_000
    SAVE_INTERVAL      = 50_000
    LOG_EVERY       = 1_000


# In[ ]:




