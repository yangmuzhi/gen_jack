"""
GAN wrapper
"""

import os
import tensorflow as tf
import tqdm
from .utils import realjack_generator, fake_z_generator

class GAN_wrapper:
    
    def __init__(self, gan, model_path=None, train_fq=100):
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        self.gan = gan
        self.train_fq = train_fq

    def train(self, epochs, batch_size=32):

        epo = tqdm.tqdm(range(epochs))
        data_gen = realjack_generator(batch_size)
        for i in epo:
            # data

            if (i+1) % self.train_fq == 0:

                z = fake_z_generator(batch_size, 7)
                jack = next(data_gen)

                for _ in range(2):
                    _, dloss = self.gan.train_disc(jack, z)

                for _ in range(6):
                    _, gloss = self.gan.train_gen(z)
                   