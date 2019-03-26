import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.utils.conv_utils import conv_output_length, deconv_length
import numpy as np
from keras.layers import Input, concatenate, Conv2D, SpatialDropout2D, UpSampling2D, LeakyReLU, Add, Lambda


def kernel_tile(input, kernel=3, stride=1):
    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='SAME')
    output_shape = output.get_shape()
    print(output_shape)
    output = tf.reshape(output, shape=[1, int(
        output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    return output


            
            


class PrimaryCapsLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(PrimaryCapsLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " filters]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.built = True

    def call(self, input_tensor, training=None):
        batch_size, height, width = 1, self.input_height, self.input_width
        
        pose = Conv2D(filters=self.num_capsule*16, 
                      kernel_size=1, 
                      strides=1, 
                      kernel_initializer=self.kernel_initializer, 
                      padding=self.padding,
                      activation=None)(input_tensor)
 
        activations = Conv2D(filters=self.num_capsule, 
              kernel_size=1, 
              strides=1, 
              kernel_initializer=self.kernel_initializer, 
              padding=self.padding,
              activation='sigmoid')(input_tensor)
    
        pose = tf.reshape(pose, shape=[-1, height, width, self.num_capsule, 16])
        activations = tf.reshape(activations, shape=[-1, height, width, self.num_capsule, 1])
        output = tf.concat([pose, activations], axis=4)
        output = tf.reshape(output, shape=[-1, height, width, self.num_capsule*17])
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.num_capsule*17)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(PrimaryCapsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
        
        
        

class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.batch_size=1

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, 16]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3] // 17

        # Transform matrix
        self.W = self.add_weight(shape=[1, self.kernel_size * self.kernel_size * self.input_num_capsule, self.num_capsule, 4, 4],
                                 initializer=self.kernel_initializer,
                                 name='W')


        self.built = True
        

    def call(self, input_tensor, training=None):
        input_tensor = kernel_tile(input_tensor, kernel=self.kernel_size, stride=self.strides)
        batch, height, width, _, _ = input_tensor.get_shape()
        output = tf.reshape(input_tensor, shape=[-1, self.kernel_size * self.kernel_size * self.input_num_capsule, 17])
        
        activation = tf.reshape(output[:, :, 16], shape=[
                                    -1, self.kernel_size * self.kernel_size * self.input_num_capsule, 1])
        
        pre_transform = tf.reshape(output[:, :, :16], shape=[-1, self.kernel_size * self.kernel_size * self.input_num_capsule, 1, 4, 4])
        print(pre_transform.get_shape())
        w = tf.tile(self.W, [-1, 1, 1, 1, 1])
        pre_transform = tf.tile(pre_transform, [1, 1, self.num_capsule, 1, 1])
        votes = tf.matmul(pre_transform, w)
        votes = tf.reshape(votes, [-1, self.kernel_size * self.kernel_size * self.input_num_capsule, self.num_capsule, 16])
        print(activation.get_shape())
        print(votes.get_shape()) # votes = 9*activation batch?
        miu, activation, _ = em_routing(self, votes, activation, self.num_capsule, routings=self.routings)
                    
        pose = tf.reshape(miu, shape=[-1, height, width, self.num_capsule, 16])
        print(pose.get_shape())
        print(activation.get_shape())
        activation = tf.reshape(activation, shape=[-1,height, width, self.num_capsule, 1])
        print(activation.get_shape())
        output = tf.reshape(tf.concat([pose, activation], axis=4), [-1, height, width, self.num_capsule*17])
        return output
    
    
    
class OutputCapsuleLayer(layers.Layer):
    def __init__(self, out_classes, **kwargs):
        super(OutputCapsuleLayer, self).__init__(**kwargs)
        self.out_classes=out_classes

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, 16]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        print(input_shape)
        self.built = True
        

    def call(self, input_tensor, training=None):
        out_caps = tf.keras.backend.expand_dims(input_tensor, axis=-1)
        out_reshaped = layers.Reshape((self.input_height, self.input_width, self.out_classes, 17))(out_caps)
        activations = Lambda(lambda out_reshaped: out_reshaped[:, :, :, :, -1])(out_reshaped)
        return activations

        
        
        
def em_routing(capslayer, votes, activation, caps_num_c, routings, regularizer=None, tag=False):
    test = []

    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])
    n_channels = int(votes.get_shape()[-1])
    epsilon = 0.00001
    ac_lambda0 = 1

    sigma_square = []
    miu = []
    activation_out = []
    print(batch_size)
    beta_v = capslayer.add_weight(shape=[caps_num_c, n_channels], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=regularizer,
                                 name='beta_v')
    beta_a = capslayer.add_weight(shape=[caps_num_c], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=regularizer,
                                 name='beta_a')
    # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
    # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
    votes_in = votes
    activation_in = activation

    for iters in range(routings):

        # e-step
        if iters == 0:
            r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
        else:
            # Contributor: Yunzhi Shi
            # log and exp here provide higher numerical stability especially for bigger number of iterations
            log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                        (tf.square(votes_in - miu) / (2 * sigma_square))
            log_p_c_h = log_p_c_h - \
                        (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
            p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

            ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + epsilon)

        # m-step
        r = r * activation_in
        r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+epsilon)

        r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
        r1 = tf.reshape(r / (r_sum + epsilon),
                        shape=[-1, caps_num_i, caps_num_c, 1])

        miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                     axis=1, keep_dims=True) + epsilon

        if iters == routings-1:
            r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
            cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                         shape=[batch_size, caps_num_c, n_channels])))) * r_sum

            activation_out = tf.nn.softmax(ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
        else:
            activation_out = tf.nn.softmax(r_sum)

    return miu, activation_out, test

