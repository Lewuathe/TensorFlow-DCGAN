from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from model import DCGAN
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_integer(
    'height', 28, "Input image height. Default is MNIST dataset"
)
tf.flags.DEFINE_integer(
    'width', 28, "Input image width. Default is MNIST data set"
)
tf.flags.DEFINE_integer(
  'epoch', 1000, 'The number of training iteration'
)

FLAGS = tf.flags.FLAGS

def main(argv):
  mnist = input_data.read_data_sets('data/mnist', one_hot=True)
  X = tf.placeholder(tf.float32, shape=[None, 784])
  z = tf.placeholder(tf.float32, shape=[None, 100])

  batch_size = 64
  model = DCGAN(FLAGS.height, FLAGS.width, X, z, batch_size)

  D_loss = model.D_loss
  G_loss = model.G_loss
  G_solver = model.G_solver
  D_solver = model.D_solver
  merged = model.merged

  with tf.Session() as sess:
    saver = tf.train.Saver()
    # if os.path.exists('./models'):
    #   saver.restore(sess, './models/dcgan.ckpt-1900')
    writer = tf.summary.FileWriter('./models', sess.graph)
    sess.run(tf.global_variables_initializer())

    for e in range(0, FLAGS.epoch):

      x, _ = mnist.train.next_batch(batch_size)
      rand = np.random.uniform(0., 1., size=[batch_size, 100])
      _, summ, dl, gl = sess.run([D_solver, merged, D_loss, G_loss], {X: x, z: rand})
      rand = np.random.uniform(0., 1., size=[batch_size, 100])
      _ = sess.run([G_solver], {z: rand})
      rand = np.random.uniform(0., 1., size=[batch_size, 100])
      _ = sess.run([G_solver], {z: rand})

      writer.add_summary(summ, global_step=e)
      if e % 100 == 0:
        saver.save(sess, './models/dcgan.ckpt', global_step=e)

    writer.close()

if __name__ == '__main__':
  tf.app.run()