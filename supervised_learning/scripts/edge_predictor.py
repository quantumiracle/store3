"""
pure vector observation based learning: position of tactip and target
task: tactip following the cylinder to reach the ball target
use 382 pins
"""

import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt

from deform_visualize import plot_list_new, plot_list_new_sim2
import pickle


class Predictor(object):
    def __init__(self, obs_dim, label_dim, ini_lr=1e-3):   
        self.hidden_dim=500  
        self.sess = tf.Session()
        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'obs')
        self.lr = tf.placeholder_with_default(ini_lr,  shape=(), name='lr')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')  # BN signal

        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        self.predict = 2*tf.layers.dense(l3, label_dim, activation=tf.nn.tanh)  # predict position and rotation
        self.loss = tf.reduce_mean(tf.square(self.predict-self.label))  # pos


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self,  batch_s, batch_label, lr, decay):
        # self.optimizer.learning_rate = lr
        # if decay:
        #     self.train_op = self.optimizer.minimize(self.loss)
        loss,_=self.sess.run([self.loss, self.train_op], {self.training: True, self.obs: batch_s, self.label: batch_label, self.lr: lr})
        # if decay: 
        #     print(self.optimizer._lr)
        return loss

    def predict_one_value(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        return predict
        
    def predict_value(self, s):
        predict = self.sess.run(self.predict, {self.obs: s})
        return predict

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver=tf.train.Saver()
        saver.restore(self.sess, path)


def plot(s):
    x=s[7::3]
    z=s[9::3]
    plot_list_new(x,z)

def state_process(s):
    factor=0.5674

    x0=s[1::3]
    z0=s[3::3]
    x=np.array(x0)/factor
    z=np.array(z0)/factor
    data=np.transpose([x,z]).reshape(-1)  # (x,y,x,y,...)

    label=s[0]

    return [label], data
