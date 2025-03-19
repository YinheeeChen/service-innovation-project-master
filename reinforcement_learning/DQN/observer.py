import numpy as np
# 存储训练过程中的reward
class Observer:
    def __init__(self):
        self.rewards = []

    def store(self, name = 'reward_list'):
        np.save(name, self.rewards)

    def add(self, value):
        self.rewards.append(value)
