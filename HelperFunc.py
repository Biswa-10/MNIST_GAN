# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:06:42 2019

@author: Biswajit
"""
import tensorflow as tf
import numpy as np 

def loss(logits_in,labels_in):
    return tf.reduce_mean(tf.square(logits_in-labels_in))

def get_next_batch(X,Y,batch_size):
    idx = np.random.randint(0,len(X),batch_size)
    batch_x = X[idx]
    batch_y = Y[idx]
    return batch_x,batch_y