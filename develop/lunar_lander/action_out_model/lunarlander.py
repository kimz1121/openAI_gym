import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import time

import keras
from keras.models       import Sequential
from keras.layers       import Dense
from keras.optimizers   import Adam

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


if __name__ == '__main__':
    env_screen = gym.make("CartPole-v1", render_mode="human")
    env =  gym.make ("CartPole-v1")

    env.reset()
    env_screen.reset()

    env.close()
    env_screen.close()

    