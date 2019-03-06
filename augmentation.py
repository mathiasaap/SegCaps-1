import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import time
from keras.preprocessing.image import *
from postprocess import oneHot2LabelMax

def elasticDeform3D(x, y, alpha, sigma, mode="constant", cval=0, is_random=False):
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    modalities = x.shape[3]
    classes = y.shape[3]

    distX = random_state.rand(*x.shape[:-1])
    distY = random_state.rand(*x.shape[:-1])
    distZ = random_state.rand(*x.shape[:-1])

    dx = gaussian_filter((distX * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((distY * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dz = gaussian_filter((distZ * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    dataChannels = []
    for modal in range(modalities):
        x_, y_, z_ = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), np.arange(x.shape[2]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1)), np.reshape(z_ + dz, (-1, 1))
        dataChannels.append(map_coordinates(x[:,:,:,modal], indices, order=1).reshape(x.shape[:-1]))

    outputChannels = []
    for c in range(classes):
        x_, y_, z_ = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), np.arange(x.shape[2]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1)), np.reshape(z_ + dz, (-1, 1))
        outputChannels.append(map_coordinates(y[:,:,:,c], indices, order=1).reshape(x.shape[:-1]))

    #x_, y_, z_ = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), np.arange(x.shape[2]), indexing='ij')
    #indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1)), np.reshape(z_ + dz, (-1, 1))

    newX = np.zeros(x.shape)
    #y = map_coordinates(y, indices, order=1).reshape(y.shape)
    for modal in range(len(dataChannels)):
        newX[:,:,:,modal] = dataChannels[modal]

    #newYValues = np.zeros(y.shape)
    #newY = np.zeros(y.shape)
    for c in range(len(outputChannels)):
        y[:,:,:,c] = outputChannels[c]


    return newX, y
    #labels rounded to preverse index property
    #return newX, np.round(y)
    
def elasticDeform2D(x, y, alpha, sigma, mode="constant", cval=0, is_random=False):
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    modalities = x.shape[2]
    classes = y.shape[2]

    distX = random_state.rand(*x.shape[:-1])
    distY = random_state.rand(*x.shape[:-1])

    dx = gaussian_filter((distX * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((distY * 2 - 1), sigma, mode=mode, cval=cval) * alpha

    dataChannels = []
    for modal in range(modalities):
        x_, y_ = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
        dataChannels.append(map_coordinates(x[:,:,modal], indices, order=1).reshape(x.shape[:-1]))

    outputChannels = []
    for c in range(classes):
        x_, y_ = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
        outputChannels.append(map_coordinates(y[:,:,c], indices, order=1).reshape(x.shape[:-1]))

    newX = np.zeros(x.shape)
    for modal in range(len(dataChannels)):
        newX[:,:,modal] = dataChannels[modal]

    for c in range(len(outputChannels)):
        y[:,:,c] = outputChannels[c]


    return newX, y

def augment_random(image, label):
    
    #happens first
    if np.random.randint(0, 5) == 3:
        image, label = elasticDeform2D(image, label, alpha=720, sigma=24, is_random=True)
    
    #Img [x,x,slices*modalities]
    num_sm = image.shape[2]
    num_classes = label.shape[2]
    img_and_mask = np.zeros((image.shape[0], image.shape[1], num_sm + num_classes))
    
    #image = np.rollaxis(image, 2, 0)
    #label = np.rollaxis(label, 2, 0)
   
    img_and_mask[:, :, :num_sm] = image
    img_and_mask[:, :, num_sm:] = label
    
    
    if np.random.randint(0,10) == 7: 
        img_and_mask = random_rotation(img_and_mask, rg=180, row_axis=0, col_axis=1, channel_axis=2,
                                       fill_mode='constant', cval=0.)


    if np.random.randint(0, 10) == 7:
        img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
                                    fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
        img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
        img_and_mask = random_zoom(img_and_mask, zoom_range=(0.8, 0.8), row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
        img_and_mask = flip_axis(img_and_mask, axis=1)

    if np.random.randint(0, 10) == 7:
        img_and_mask = flip_axis(img_and_mask, axis=0)
      
    
    image = img_and_mask[:, :, :num_sm] 
    label = img_and_mask[:, :, num_sm:]
    
    #image = np.rollaxis(image, 0, 3)
    #label = np.rollaxis(label, 0, 3)
    
    default_one_hot_label = [0] * num_classes
    default_one_hot_label[0] = 1
    label[np.where(np.sum(label, axis =-1) == 0)] = default_one_hot_label
    
    label = oneHot2LabelMax(label)
    label = np.eye(num_classes)[label]
   
    
    
            
    #Label [x,x,classes]
    
    # img = [x,y,slices*modalities]
    # [x,y,classes]
    
    # img_and_mask = [slices*modalities + classes, x, y]
        
    return image, label