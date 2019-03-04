import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import time

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
    rot = [random.randint(0, 3) for i in range(3)]
    flip = [random.randint(0, 1) for i in range(3)]
    '''if rot[0]:
        image = np.rot90(image, rot[0], axes=(1, 2))
        label = np.rot90(label, rot[0], axes=(1, 2))
    if rot[1]:
        image = np.rot90(image, rot[1], axes=(0, 2))
        label = np.rot90(label, rot[1], axes=(0, 2))'''
    if rot[2]:
        image = np.rot90(image, rot[2], axes=(0, 1))
        label = np.rot90(label, rot[2], axes=(0, 1))

    if flip[0]:
        image = image[::-1]
        label = label[::-1]
    if flip[1]:
        image = image[:, ::-1]
        label = label[:, ::-1]
    if flip[2]:
        image = image[:, :, ::-1]
        label = label[:, :, ::-1]
        
    #image, label = elasticDeform3D(image, label, alpha=720, sigma=24, mode='reflect', is_random=True)
        
    return image, label