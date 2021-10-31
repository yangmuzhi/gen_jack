"""
GAN for jackspot
"""

import tensorflow as tf
from .base import Base
import numpy as np

class GAN(Base):

    def __init__(self, save_path, obs_dims=7, z_dims=7, glr=1e-5, dlr=1e-4, C=10):


        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, obs_dims])
        self.C = C
        # with tf.variable_scope('discriminator'):
        self.scope = tf.get_variable_scope().name
        self.real_jack = tf.placeholder(dtype=tf.float32, shape=[None, obs_dims])
        # add noise for stabilise training
        # expert += tf.random_normal(tf.shape(expert), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2

        # noise
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, z_dims])
        self.fake_jack = self.generator_network(obs=self.z)
        # add noise for stabilise training
        # agent += tf.random_normal(tf.shape(agent), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2

        with tf.variable_scope('network') as network_scope:
            self.disc_prob = self.discriminator_network(obs=self.real_jack)
            network_scope.reuse_variables()  # share parameter
            self.gen_prob = self.discriminator_network(obs=self.fake_jack)

        with tf.variable_scope('gloss'):
            self.reg = tf.reduce_mean(tf.where(tf.greater_equal(self.fake_jack, [33,33,33,33,33,33,16]), 
                                      tf.square(self.fake_jack-[33,33,33,33,33,33,16]), tf.where(tf.greater_equal(np.array([1,1,1,1,1,1,1], dtype=np.float32).reshape(-1, 7), 
                                      self.fake_jack), tf.square(self.fake_jack-[1,1,1,1,1,1,1]), self.fake_jack-self.fake_jack)))
            # self.reg = 0 
            self.gloss = tf.reduce_mean(1-tf.log(self.gen_prob)) + self.C * self.reg
         
        with tf.variable_scope('dloss'):
            loss_disc = tf.reduce_mean(tf.log(tf.clip_by_value(self.disc_prob, 0.01, 1)))
            loss_gen = tf.reduce_mean(tf.log(tf.clip_by_value(1 - self.gen_prob, 0.01, 1))) 
            loss = loss_disc + loss_gen
            self.dloss = -loss + self.C * self.reg

        g_optimizer = tf.train.RMSPropOptimizer(glr)
        g_optimizer = tf.train.RMSPropOptimizer(dlr)
        self.disc_train_op = g_optimizer.minimize(self.dloss)
        self.gen_train_op = g_optimizer.minimize(self.gloss)
        # self.prob = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent
        
        super().__init__(save_path=save_path, rnd=1234)

    def discriminator_network(self, obs):
        layer_1 = tf.layers.dense(inputs=obs, units=64, activation=tf.nn.leaky_relu, name='layer1_d')
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.leaky_relu, name='layer2_d')
        prob = tf.layers.dense(inputs=layer_2, units=1, activation=tf.sigmoid, name='prob')
        return prob
    
    def generator_network(self, obs):
        layer_1 = tf.layers.dense(inputs=obs, units=64, activation=tf.nn.leaky_relu, name='layer1_g')
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.leaky_relu, name='layer2_g')
        layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=None, name='layer3_g')
        jack = tf.layers.dense(inputs=layer_3, units=7, activation=None, name='jack_g')*33
        # jack = tf.clip_by_value(jack, 1, 33)

        return jack

    def train_disc(self, real_jack, z):
        return self.sess.run([self.disc_train_op, self.dloss], feed_dict={self.real_jack: real_jack,
                                                            self.z: z})

    def train_gen(self, z):
        return self.sess.run([self.gen_train_op, self.gloss], feed_dict={self.z: z})
    
    def get_jack(self, z):
        return self.sess.run(self.fake_jack, feed_dict={self.z: z})

    def get_prob(self, fake_jack):
        return self.sess.run(self.gen_prob, feed_dict={self.fake_jack: fake_jack})

def realjack_generator(real_jack, batch_size):

    np.random.shuffle(real_jack)
    jack = real_jack
    
    while True:
        if batch_size <= jack.shape[0]:
            res = jack[0:batch_size]
            jack = jack[batch_size:]
        else:
            np.random.shuffle(real_jack)
            jack = real_jack
        yield res

def fake_z_generator(batch_size, dims):

    return np.random.normal(loc=1, scale=3, size=[batch_size,dims])
