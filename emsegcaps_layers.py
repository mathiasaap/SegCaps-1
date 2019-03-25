import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.utils.conv_utils import conv_output_length, deconv_length
import numpy as np


def kernel_tile(input, kernel=3, stride=1):
    # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')

    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(
        output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    return output

class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, 16]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, 4, 4],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, input_tensor, training=None):
        input_tensor = kernel_tile(input_tensor, kernel=self.kernel_size, stride=self.strides)
        
        batch_size, height, width, _ = input_tensor.get_shape()
        input_tensor = tf.reshape(input_tensor, shape=[batch_size *
                                   height * width, self.kernel_size * self.kernel_size * self.num_capsule, 17])
        
