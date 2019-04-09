from keras import layers, models
from keras.layers import Lambda
from keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

from emsegcaps_layers import PrimaryCapsLayer, ConvCapsuleLayer, DeconvCapsuleLayer, OutputCapsuleLayer


def SegCapsEM(input_shape, modalities=1, n_class=2):
    
    split_capsule = Lambda(lambda input_tensor: tf.reshape(input_tensor, 
                       shape=[1, 
                       input_tensor.get_shape()[1],
                       input_tensor.get_shape()[2], 
                       input_tensor.get_shape()[3]//17, 
                       17]))
    merge_capsule = Lambda(lambda input_tensor: tf.reshape(input_tensor, 
                       shape=[1, 
                       input_tensor.get_shape()[1],
                       input_tensor.get_shape()[2], 
                       input_tensor.get_shape()[3]*17]))
    
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1')(x)
    
    # Layer 1: Primary Capsule: Conv cap with routing 1
    #primary_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=n_class, strides=1, padding='same',
    #                                routings=1, name='primarycaps')(conv1_reshaped)
    
    primary_caps = PrimaryCapsLayer(kernel_size=1, 
                                    num_capsule=1, 
                                    strides=1, 
                                    padding='same', 
                                    name='primarycaps')(conv1)
    
    conv_caps_0_1 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=2, 
                                    strides=1, 
                                    padding='same',
                                    name='conv_caps_0_1')(primary_caps)
    
    conv_caps_1_1 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=4, 
                                    strides=2, 
                                    padding='same',
                                    name='conv_caps_1_1')(conv_caps_0_1)
    conv_caps_1_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=4, 
                                    strides=1, 
                                    padding='same',
                                    name='conv_caps_1_2')(conv_caps_1_1)
    
    conv_caps_2_1 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=8, 
                                    strides=2, 
                                    padding='same',
                                    name='conv_caps_2_1')(conv_caps_1_2)
    
    conv_caps_2_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=8, 
                                    strides=1, 
                                    padding='same',
                                    name='conv_caps_2_2')(conv_caps_2_1)
    
    conv_caps_3_1 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=16, 
                                    strides=2, 
                                    padding='same',
                                    name='conv_caps_3_1')(conv_caps_2_2)
    
    conv_caps_3_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=16, 
                                    strides=1, 
                                    padding='same',
                                    name='conv_caps_3_2')(conv_caps_3_1)
    

    
   
    
    deconv_caps_1_1 = DeconvCapsuleLayer(kernel_size=3, 
                                    num_capsule=8, 
                                    strides=2, 
                                    padding='same',
                                    name='deconv_caps_1_1')(conv_caps_3_2)
    

    conv_caps_2_2_reshaped = split_capsule(conv_caps_2_2)
    deconv_caps_1_1_reshaped = split_capsule(deconv_caps_1_1)
    skip_1 = layers.Concatenate(axis=-2, name='skip_1')([conv_caps_2_2_reshaped, deconv_caps_1_1_reshaped])
    skip_1_reshaped = merge_capsule(skip_1)
    
    deconv_caps_1_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=8, 
                                    strides=1, 
                                    padding='same',
                                    name='deconv_caps_1_2')(skip_1_reshaped)
    
    deconv_caps_2_1 = DeconvCapsuleLayer(kernel_size=3, 
                                    num_capsule=4, 
                                    strides=2, 
                                    padding='same',
                                    name='deconv_caps_2_1')(conv_caps_2_2)
    
    conv_caps_1_2_reshaped = split_capsule(conv_caps_1_2)
    deconv_caps_2_1_reshaped = split_capsule(deconv_caps_2_1)
    skip_2 = layers.Concatenate(axis=-2, name='skip_2')([conv_caps_1_2_reshaped, deconv_caps_2_1_reshaped])
    skip_2_reshaped = merge_capsule(skip_2)
    
    deconv_caps_2_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=4, 
                                    strides=1, 
                                    padding='same',
                                    name='deconv_caps_2_2')(skip_2_reshaped)
    
    deconv_caps_3_1 = DeconvCapsuleLayer(kernel_size=3, 
                                    num_capsule=2, 
                                    strides=2, 
                                    padding='same',
                                    name='deconv_caps_3_1')(deconv_caps_2_2)
    
    conv_caps_0_1_reshaped = split_capsule(conv_caps_0_1)
    deconv_caps_3_1_reshaped = split_capsule(deconv_caps_3_1)
    skip_3 = layers.Concatenate(axis=-2, name='skip_3')([conv_caps_0_1_reshaped, deconv_caps_3_1_reshaped])
    skip_3_reshaped = merge_capsule(skip_3)
    
    deconv_caps_3_2 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=2, 
                                    strides=1, 
                                    padding='same',
                                    name='deconv_caps_3_2')(skip_3_reshaped)
    
    
    #print(deconv_caps_3_2.get_shape())
    #assert False
    out_caps = OutputCapsuleLayer(out_classes=n_class,
                                  name='out_seg')(deconv_caps_3_2)
                                    
                             
    train_model = models.Model(inputs=x, outputs=out_caps)


    return [train_model]
