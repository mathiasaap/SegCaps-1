from keras import layers, models
from keras import backend as K
K.set_image_data_format('channels_last')

from emsegcaps_layers import ConvCapsuleLayer

def SegCapsEM(input_shape, modalities=1, n_class=2):    
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)


    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    
    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=n_class, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

   

    out_caps = primary_caps
    train_model = models.Model(inputs=[x, y], outputs=out_caps)


    return [train_model]
