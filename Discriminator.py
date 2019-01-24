# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:56:59 2019

@author: Biswajit
"""
import tensorflow as tf

class Discriminator:
    
    def __init__(self,img_shape):
        self.reuse = None
        self.rows,self.cols,self.channels = img_shape
        
        
    def discriminatorFn(self,X,y,reuse=None):
        
        with tf.variable_scope('dis',reuse=tf.AUTO_REUSE):
            
            hidden1 = tf.layers.dense(inputs=X,units=256,activation = tf.nn.leaky_relu)
            y_hidden1 = tf.layers.dense(inputs=y,units=32,activation=tf.nn.leaky_relu)
        
            hidden2 = tf.layers.dense(inputs=tf.concat((hidden1,y_hidden1),axis = 1),units=32,activation=tf.nn.leaky_relu)
            
            logits = tf.layers.dense(hidden2,units=1)
            output = tf.sigmoid(logits)
    
            return output, logits