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
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
from deform_visualize import plot_list_new, plot_list_new_sim2
import pickle


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--rotat_test', dest='rotat_test', action='store_true', default=False)

args = parser.parse_args()


class Classifier(object):
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
        self.predict = 2*np.pi*tf.layers.dense(l3, label_dim, activation=tf.nn.sigmoid)  # predict position and rotation
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

def state_process(s, s0):
    dis_threshold=0.05
    x0=s[1::3]
    z0=s[3::3]
 
    x=s[1::3]
    z=s[3::3]
    dis=np.abs(x-x0)+np.abs(z-z0)
    dis_threshold = max(1.2*np.average(dis), dis_threshold)
    dis_idx=np.argwhere(dis>dis_threshold).reshape(-1)
    dis[dis<=dis_threshold]=0
    dis[dis>dis_threshold]=1
    # print(dis_idx)
    # plt.figure(figsize=(5,4))  # this line cause memory keeping increasing if not close the figure

    # plot_list_new(x-object_x, z-object_z, dis_idx)  
    processed_state=dis
    label=s[0]
    return [label], processed_state

# def state_process(s):
#     factor=0.5674

#     x0=s[1::3]
#     z0=s[3::3]
#     x=np.array(x0)/factor
#     z=np.array(z0)/factor
#     data=np.transpose([x,z]).reshape(-1)  # (x,y,x,y,...)

#     label=s[0]

#     return [label], data


def Predict(input, model_path = './model/class_obj'): 
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 2  # 2 position
    lr=2e-2
    classifier = Classifier(obs_dim, state_dim, lr)
    classifier.load(model_path) 
    predict = classifier.predict_one_value(input)
    return predict

if __name__ == '__main__':
    model_path = './model/comparison/random0.2/class_obj'
    training_episodes = 80000
    input_dim = 91  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    output_dim = 1
    lr=1e-3
    decay=0 # decay signal of lr
    classifier = Classifier(input_dim, output_dim, lr)

    if args.train:
        data_file=open('compare_data/raw_data02.pickle', "rb")
        raw_data=pickle.load(data_file)
        data=[]
        label=[]
        s0=raw_data[0]
        for i in range(1,len(raw_data)):
            s=raw_data[i]
            label_i, data_i=state_process(s, s0)
            ''' add noise '''
            data_i=data_i+np.random.normal(0, 1e-2, data_i.shape)  
            data.append(data_i)
            label.append(label_i)
        loss_list=[]
        # classifier.load(model_path)

        for eps in range(training_episodes):
            if eps%40000==0 and eps>1:
                lr *=0.5
                decay=1
            else:
                decay=0
            loss = classifier.train(data, label, lr, decay)
            if eps==0:
                loss_list.append(loss)
            else:
                loss_list.append(0.9*loss_list[-1]+0.1*loss)
            print('Eps: {}, Loss: {}'.format(eps, loss))
            if eps % 100 ==0:
                plt.yscale('log')
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('classify_trainwithdataobj2.png')
                classifier.save(model_path)
        
        np.savetxt('trainwithdata.txt', np.array(loss_list)[:, np.newaxis], fmt='%.4f', newline=', ')
        round_loss_list=list(np.around(np.array(loss_list),4))
        print(round_loss_list)




# test with testing dataset, all at once
    if args.test:
        test_data_file=open('data/raw_data.pickle', "rb")
        raw_data=pickle.load(test_data_file)
        data=[]
        label=[]
        classifier.load(model_path)  
        for i in range(80):
            s=raw_data[i]
            label_i, data_i=state_process(s)
            print(label_i)
            data.append(data_i)
            label.append(label_i)
            predict = classifier.predict_one_value(data_i)[0]
            print(predict)
            xy=data_i.reshape(-1,2)
            # plot_list_new_sim2(xy,i,predict, label_i)
            print(i)
