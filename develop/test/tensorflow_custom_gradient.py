import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import time

import tensorflow as tf
import keras
from keras.models       import Sequential
from keras.layers       import Dense
from keras.optimizers   import Adam
from keras.optimizers   import SGD

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# print(tf.executing_eagerly())


@tf.custom_gradient
def log1pexp(x):
    print("hello")
    e = tf.exp(x)
    def grad(upstream):
        return upstream * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad # 실행가능한 함수핸들 2개 Loss, Loss_gradient 를 반환하여야 한다.

class MeanSquaredError_custom(tf.keras.losses.Loss):
    # def __init__(self):
    #     super.__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
  
    

# loss_instance = MeanSquaredError_custom()

optimizer_custom = tf.keras.optimizers.SGD()
optimizer_custom.apply_gradients(log1pexp)


model = keras.models.Sequential()
model.add(keras.Input(shape=(1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss=MeanSquaredError_custom(), optimizer=tf.keras.optimizers.SGD())
# model.compile(loss='mse', optimizer=Adam())
# 커스텀 그래디언트 적용까지는 다음기회에....

# 먼저.... cartpole 부터 도전??
# Lunarlander 도전?
# 카래이싱도 좋을 듯... 그러나 이건 오늘 안에 끝내기는 무리??

# cartpole 되는 것 보고 결정해야 할 듯.



tf.custom_gradient()

data = tf.constant(3, dtype="float")
log1pexp(data)
