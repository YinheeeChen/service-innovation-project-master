import numpy as np
import matplotlib.pyplot as plt
from dqn import DQN
import torch
import pdb

# 类Epsilon用于控制epsilon的变化
# epsilon初始值为1.0，最终值为0.1，探索帧数为1000000，总帧数为30000000
# 通过step方法，每次调用epsilon.step()，epsilon的值会根据当前帧数变化
# 通过gen_action方法，可以根据当前的epsilon值，生成一个动作
# 通过show方法，可以显示当前的epsilon值
# 通过get方法，可以获取当前的epsilon值
# 通过self_act方法，可以根据当前的epsilon值，生成一个动作，并且可以选择是否显示Q值
# 通过show_Qs方法，可以显示Q值
class Epsilon:
    def __init__(self, initial_eps = 1.0, final_eps = 0.1, explore_frames = 1_000_000, total_frames = 30_000_000, action_space_n = 4):
        self.epsilon = initial_eps
        self.slope = (final_eps - initial_eps) / explore_frames
        self.action_space = action_space_n
        self.final_eps = final_eps
        self.second_slope = (-final_eps) / (total_frames - explore_frames)

    def step(self):
        if self.epsilon > self.final_eps:
            self.epsilon += self.slope
        elif self.epsilon > 0:
            self.epsilon += self.second_slope
        return self

    def gen_action(self, state, dqn:DQN, render = False):
        epsilon = self.epsilon
        if np.random.random() > epsilon:
            qs = dqn.predict(state/255.0)
            if render:
                print('Q values: {}'.format(qs))
            return int(torch.argmax(qs[0]))
        return np.random.randint(0, self.action_space)

    def show_Qs(self, qs):
        plt.clf()
        plt.title('Q-values')
        plt.ylim(-1, 9)
        plt.bar(np.arange(len(qs)), qs)
        plt.draw()
        plt.pause(1e-17)

    def self_act(self, state, dqn, epsilon, render = False):
        if np.random.random() > epsilon:
            qs = dqn.predict(state / 255.0)
            if render:
                self.show_Qs(qs.detach().numpy()[0])
                print('Q-values: {}'.format(qs))
            return int(torch.argmax(qs))
        return np.random.randint(0, self.action_space)

    def get(self):
        return self.epsilon

    def show(self):
        print(self.epsilon)
        return self


def test():
    e = Epsilon()
    for i in range(2_000_000):
        e.step().show()

# test()

