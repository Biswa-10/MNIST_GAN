# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 05:02:12 2019

@author: Biswajit
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:32:37 2019

@author: Biswajit
"""

import tensorflow as tf

class Generator:

    
    def __init__(self,img_shape,z_shape):

        self.z_shape=z_shape
        self.rows,self.cols,self.channels = img_shape
        self.reuse = None
                   
    def generatorFn(self,z,y,reuse=None):

        with tf.variable_scope('gen',reuse=reuse):

            hidden1 = tf.layers.dense(inputs=z,units=64,activation = tf.nn.leaky_relu)
            y_hidden1 = tf.layers.dense(inputs=y,units=32,activation=tf.nn.leaky_relu)

            hidden2 = tf.layers.dense(inputs=tf.concat((hidden1,y_hidden1),axis = 1),units=392,activation=tf.nn.leaky_relu)

            output = tf.layers.dense(hidden2,units=784,activation=tf.nn.tanh)

            return output
