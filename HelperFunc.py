# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:06:42 2019

@author: Biswajit
"""
import tensorflow as tf
import numpy as np 

def loss(labels,y):
    return tf.reduce_mean(tf.square(y-labels))


def get_next_batch(X,Y,batch_size):
    idx = np.random.randint(0,len(X),batch_size)
    batch_x = X[idx]
    batch_y = Y[idx]
    return batch_x,batch_y

