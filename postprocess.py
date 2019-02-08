import numpy as np
def oneHot2Label(array):
    classes = array.shape[-1]
    classaxis = len(array.shape) - 1
    return np.argmin(array, axis = classaxis)
def oneHot2LabelMax(array):
    classes = array.shape[-1]
    classaxis = len(array.shape) - 1
    return np.argmax(array, axis = classaxis)

def oneHot2LabelMin(array):
    classes = array.shape[-1]
    classaxis = len(array.shape) - 1
    return np.argmin(array, axis = classaxis)