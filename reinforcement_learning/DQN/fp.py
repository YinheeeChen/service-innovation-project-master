import cv2
import numpy as np
import pdb


# def preprocess(image):
#     image = cv2.resize(image, (84, 110))
#     image = image[26:, :]
#     image = np.mean(image, axis=2)
#     return image.astype(np.uint8)

# 对图像进行预处理
def preprocess(image):
    # print(type(image))  # 打印图像的类型
    # print(image)  # 打印图像的内容

    # pdb.set_trace()

    return np.mean(image[26:, :], axis=2, dtype=np.uint8)


# 测试预处理函数
def test_preprocess():
    import gymnasium as gym
    env = gym.make('BreakoutDeterministic-v4')
    state, info = env.reset()
    state = preprocess(state)
    print(state.shape)
    cv2.imshow('t', state)
    cv2.waitKey(1)
    print(state.shape)
    assert state.shape == (84, 84), 'shape not right'


# 测试预处理函数的速度
def timeit():
    import time
    import gymnasium as gym
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    n = 1000
    states = [env.step(0)[0] for _ in range(n)]
    t = time.time()
    for state in states:
        preprocess(state)
    print((time.time() - t) / n)

# test_preprocess()
# input()
# timeit()
