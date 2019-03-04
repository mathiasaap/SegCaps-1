from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, SpatialDropout2D, UpSampling2D, LeakyReLU, Add, Lambda
from keras.activations import softmax, sigmoid
from keras import regularizers
import tensorflow as tf

def createResidualBlock(input, filters, level):
    layer = LeakyReLU(alpha = 0.01)(input)
    layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(layer)
    layer = SpatialDropout2D(rate = 0.3, data_format='channels_last')(layer)
    layer = LeakyReLU(alpha = 0.01)(layer)
    
    layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(layer)

    added = Add()([input, layer])
    output = LeakyReLU(alpha = 0.01)(added)
    return output

def downsampleBlock(inputLayer, filters, stride, level):
    layer = Conv2D(filters, (3, 3), strides = stride, kernel_initializer='he_normal', padding='same')(inputLayer)
    return layer

def upsampleBlock(input, filters, level):
    upscale = UpSampling2D(2)(input)
    layer = Conv2D(filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(upscale)
    layer = LeakyReLU(alpha = 0.01)(layer)
    return layer

def localizationModule(input, level, filters):
    layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(input)
    
    ds = LeakyReLU(alpha = 0.01)(layer)
    layer = Conv2D(filters = filters//2, kernel_size = 1, strides = 1, padding = 'same',  kernel_initializer = 'he_normal')(ds)
    
    layer = LeakyReLU(alpha = 0.01)(layer)
    return layer, ds

def segmentationLayer(inputLayer, out_classes):
    layer = Conv2D(filters = out_classes, kernel_size = 1, strides = 1, kernel_initializer = 'he_normal', padding = 'same')(inputLayer)
    return layer

def pixel_wise_softmax(output_map):
    max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
    exponential_map = tf.exp(output_map - max_axis)
    normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
    return exponential_map / normalize

def ResidualUnet2D(input_shape, out_classes):
    #with tf.variable_scope("resunet", reuse = False):
    x = Input(input_shape)
    conv = downsampleBlock(x, 16, 1, 1)
    layer1 = createResidualBlock(conv, 16, 1)

    conv = downsampleBlock(layer1, 32, 2, 2)
    layer2 = createResidualBlock(conv, 32, 2)

    conv = downsampleBlock(layer2, 64, 2, 3)
    layer3 = createResidualBlock(conv, 64, 3)

    conv = downsampleBlock(layer3, 128, 2, 4)
    layer4 = createResidualBlock(conv, 128, 4)

    conv = downsampleBlock(layer4, 256, 2, 5)
    layer5 = createResidualBlock(conv, 256, 5)

    layer6 = upsampleBlock(layer5, 128, 6)
    layer6 = Conv2D(filters= 128, kernel_size = 1, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(layer6)

    layer6 = LeakyReLU(alpha = 0.01)(layer6)

    concatenated = concatenate([layer4, layer6], axis = 3)
    local7, ds1 = localizationModule(concatenated, 7, 256)
    layer7 = upsampleBlock(local7, 64, 7)
    concatenated = concatenate([layer3, layer7], axis = 3)
    local8, ds2 = localizationModule(concatenated, 8, 128)
    layer8 = upsampleBlock(local8, 32, 8)
    segmentationLayer8 = segmentationLayer(ds2, out_classes)
    upscale8 = UpSampling2D(2)(segmentationLayer8)

    concatenated = concatenate([layer2, layer8], axis = 3)
    local9, ds3 = localizationModule(concatenated, 9, 64)
    layer9 = upsampleBlock(local9, 16, 9)
    segmentationLayer9 = segmentationLayer(ds3, out_classes)
    upscale9 = UpSampling2D(2)(Add()([upscale8, segmentationLayer9]))

    concatenated = concatenate([layer1, layer9], axis = 3)
    layer10 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(concatenated)

    layer10 = LeakyReLU(alpha = 0.01)(layer10)
    segmentationLayer10 = segmentationLayer(layer10, out_classes)

    out = Add()([segmentationLayer10, upscale9])

    
    def createSoftmax(out):
        return softmax(out, axis = 3)
        #return pixel_wise_softmax(out)
    def createSigmoid(out):
        return sigmoid(out)
    
    #ONLY IF DICE
    if out_classes == 1: 
        out = Lambda(createSigmoid)(out) # Does this work?

    model = Model(inputs=[x], outputs=[out])
    return model