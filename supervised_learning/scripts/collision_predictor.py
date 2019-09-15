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


class Predictor(object):
    def __init__(self, obs_dim, label_dim, ini_lr, num_obj=3):   
        self.hidden_dim=500  
        self.num_obj=num_obj        
        self.sess = tf.Session()
        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.label_obj = tf.placeholder(tf.int8, [None, num_obj], 'label_obj')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'obs')
        self.lr = tf.placeholder_with_default(ini_lr,  shape=(), name='lr')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')  # BN signal

        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        self.predict = tf.layers.dense(l3, label_dim)  # predict position and rotation
        logits = tf.layers.dense(l2, self.num_obj, tf.nn.relu)
        self.predict_obj = tf.nn.softmax(logits)  # predict index of object
        self.loss1 = tf.reduce_mean(tf.square(self.predict[:, 3:]-self.label[:, 3:]))  # pos

        self.loss = self.loss1
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self,  batch_s, batch_label, batch_label_obj, lr, decay):
        # self.optimizer.learning_rate = lr
        # if decay:
        #     self.train_op = self.optimizer.minimize(self.loss)
        loss,_=self.sess.run([self.loss, self.train_op], {self.training: True, self.obs: batch_s, self.label: batch_label, self.label_obj: batch_label_obj, self.lr: lr})
        # if decay: 
        #     print(self.optimizer._lr)
        return loss

    def predict_one_value(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return np.concatenate((predict_obj, predict), axis=1)
        
    def predict_value(self, s):
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return np.concatenate((predict_obj, predict), axis=1)

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
    ps=np.concatenate((s[7::3],s[9::3]))   # ((x,x,x..),(y,y,y,...))
    return ps


def to_one_hot(idx_list): # return one-hot vector list for object index predicting
    num_samples = len(idx_list)
    # print(idx_list.shape)
    # print(num_samples, self.num_obj)
    one_hot = np.zeros((num_samples, num_obj))
    one_hot[np.arange(num_samples), np.array(idx_list)] = 1

    return one_hot


def Predict(input, model_path = './model/class_obj'): 
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)
    classifier.load(model_path) 
    predict = classifier.predict_one_value(input)
    return predict

