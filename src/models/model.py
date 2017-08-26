import tensorflow as tf


class DCGAN():
  def __init__(self, height, width, X, z, batch_size):
    '''

    :param height:
    :param width:
    :param X: input tensor [batch_size, height, width, channel]
    :param z: input random value [batch_size, length]
    '''
    self.height = height
    self.width = width
    self.batch_size = batch_size

    self.G = self.generator(z, 'generator')

    tf.summary.image('sample_image', self.G)
    D_real = self.discriminator(X, False, 'discriminator')
    D_fake = self.discriminator(self.G, True, 'discriminator')

    D_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real), name='D_loss_real'),
      name='D_loss_real')
    D_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake), name='D_loss_fake'),
      name='D_loss_fake')
    self.G_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake), name='G_loss'),
      name='G_loss')

    self.D_loss = D_loss_fake + D_loss_real

    tf.summary.scalar('D_loss_real', D_loss_real)
    tf.summary.scalar('D_loss_fake', D_loss_fake)
    tf.summary.scalar('D_loss', self.D_loss)
    tf.summary.scalar('G_loss', self.G_loss)

    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('discriminator')]
    g_params = [v for v in vars if v.name.startswith('generator')]

    self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.1).minimize(self.D_loss, var_list=d_params)
    self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.3).minimize(self.G_loss, var_list=g_params)

    self.merged = tf.summary.merge_all()

  def conv2d(self, input, output_dim, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):
    with tf.variable_scope(name):
      W = tf.get_variable('conv2d_w', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
      b = tf.get_variable('conv2d_b', [output_dim], initializer=tf.zeros_initializer())
      tf.summary.histogram('conv2d_w', W)
      tf.summary.histogram('conv2d_b', b)

    return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

  def deconv2d(self, input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
    with tf.variable_scope(name):
      W = tf.get_variable('deconv2d_w', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
      b = tf.get_variable('deconv2d_b', [output_dim], initializer=tf.zeros_initializer())
      tf.summary.histogram('deconv2d_w', W)
      tf.summary.histogram('deconv2d_b', b)

      input_shape = input.get_shape().as_list()
      output_shape = [batch_size, int(input_shape[1] * strides[0]), int(input_shape[2] * strides[1]), output_dim]

      deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, strides[0], strides[1], 1])

      return deconv + b

  def dense(self, input, output_dim, stddev=0.2, name='dense'):
    with tf.variable_scope(name):
      shape = input.get_shape()
      W = tf.get_variable('dense_w', [shape[1], output_dim], tf.float32,
                          initializer=tf.random_normal_initializer(stddev=stddev))
      b = tf.get_variable('dense_b', [output_dim], initializer=tf.zeros_initializer())
      tf.summary.histogram('dense_w', W)
      tf.summary.histogram('dense_b', b)

      return tf.matmul(input, W) + b

  def batch_norm(self, input, name='batch_norm'):
    with tf.variable_scope(name):
      output_dim = input.get_shape()[-1]
      beta = tf.get_variable('batch_norm_beta', [output_dim], initializer=tf.zeros_initializer())
      gamma = tf.get_variable('batch_norm_gamma', [output_dim], initializer=tf.ones_initializer())

      if len(input.get_shape()) == 2:
        mean, var = tf.nn.moments(input, [0])
      else:
        mean, var = tf.nn.moments(input, [0, 1, 2])
      return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

  def leaky_relu(self, input, leaky=0.2, name='leaky_relu'):
    return tf.maximum(input, leaky * input)

  def discriminator(self, X, reuse=False, name='discriminator'):
    '''
    Discriminator is trained for image classification which discriminate
    generative image and real image.
    :param X:
    :param reuse:
    :param name:
    :return:
    '''
    with tf.variable_scope(name, reuse=reuse):
      # if len(X.get_shape()) > 2:
      #   d_conv1 = self.conv2d(X, output_dim=64, name='d_conv1')
      # else:
      d_reshaped = tf.reshape(X, [-1, self.height, self.width, 1])
      d_conv1 = self.conv2d(d_reshaped, output_dim=64, name='d_conv1')

      d_h1 = self.leaky_relu(d_conv1)
      d_conv2 = self.conv2d(d_h1, output_dim=128, name='d_conv2')
      d_h2 = self.leaky_relu(d_conv2)
      d_r2 = tf.reshape(d_h2, [self.batch_size, 7 * 7 * 128], name='d_r2')
      d_h3 = self.leaky_relu(d_r2)
      d_h4 = tf.nn.dropout(d_h3, 0.5)
      d_h5 = self.dense(d_h4, output_dim=1, name='d_h5')
      return tf.nn.sigmoid(d_h5)

  def generator(self, z, name='generator'):
    '''
    Generator is trained for generating image according to input image space.
    z is a seed random value to generate sample image.
    :param z:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
      g_1 = self.dense(z, output_dim=1024, name='g_1')
      g_bn1 = self.batch_norm(g_1, name='g_bn1')
      g_h1 = tf.nn.relu(g_bn1)
      g_2 = self.dense(g_h1, output_dim=7 * 7 * 128, name='g_2')
      g_bn2 = self.batch_norm(g_2, name='g_bn2')
      g_h2 = tf.nn.relu(g_bn2)
      g_r2 = tf.reshape(g_h2, [-1, 7, 7, 128])
      g_dconv1 = self.deconv2d(g_r2, output_dim=64, batch_size=self.batch_size, name='g_deconv1')
      g_bn3 = self.batch_norm(g_dconv1, name='g_bn3')
      g_h3 = tf.nn.relu(g_bn3)
      g_dconv2 = self.deconv2d(g_h3, output_dim=1, batch_size=self.batch_size, name='g_dconv2')
      g_r3 = tf.reshape(g_dconv2, [-1, self.height, self.width, 1])
      return tf.nn.sigmoid(g_r3)
