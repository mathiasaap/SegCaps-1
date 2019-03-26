from keras import layers, models
from keras.layers import Lambda
from keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

from emsegcaps_layers import PrimaryCapsLayer, ConvCapsuleLayer, OutputCapsuleLayer

def SegCapsEM(input_shape, modalities=1, n_class=2):    
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
    
    # Layer 1: Primary Capsule: Conv cap with routing 1
    #primary_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=n_class, strides=1, padding='same',
    #                                routings=1, name='primarycaps')(conv1_reshaped)
    
    primary_caps = PrimaryCapsLayer(kernel_size=1, 
                                    num_capsule=1, 
                                    strides=1, 
                                    padding='same', 
                                    name='primarycaps')(conv1)
    
    conv_caps_1 = ConvCapsuleLayer(kernel_size=3, 
                                    num_capsule=2, 
                                    strides=1, 
                                    padding='same',
                                    name='conv_caps_1')(primary_caps)
    conv_caps_2 = ConvCapsuleLayer(kernel_size=3, 
                                num_capsule=4, 
                                strides=1, 
                                padding='same',
                                name='conv_caps_2')(conv_caps_1)
    conv_caps_n = ConvCapsuleLayer(kernel_size=3, 
                            num_capsule=2, 
                            strides=1, 
                            padding='same',
                            name='conv_caps_n')(conv_caps_2)

    
    
    out_caps = OutputCapsuleLayer(out_classes=n_class,
                                  name='out_seg')(conv_caps_n)
                                    
                             
    train_model = models.Model(inputs=x, outputs=out_caps)


    return [train_model]
