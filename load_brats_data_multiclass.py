'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.
'''

from __future__ import print_function

import threading
from os.path import join, basename
from os import mkdir
from glob import glob
import csv
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import rand, shuffle
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import time

from scipy import linalg
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from keras.preprocessing.image import *

from custom_data_aug import elastic_transform, salt_pepper_noise

debug = 0

def load_data(root, split):
    # Load the training and testing lists
    with open(join(root, 'split_lists', 'train_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        training_list = list(reader)

    with open(join(root, 'split_lists', 'test_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        testing_list = list(reader)

    new_training_list, validation_list = train_test_split(training_list, test_size = 0.1, random_state = 7)
    if new_training_list == []: # if training_list only have 1 image file.
        new_training_list = validation_list
    return new_training_list, validation_list, testing_list

def compute_class_weights(root, train_data_list):
    '''
        We want to weight the the positive pixels by the ratio of negative to positive.
        Three scenarios:
            1. Equal classes. neg/pos ~ 1. Standard binary cross-entropy
            2. Many more negative examples. The network will learn to always output negative. In this way we want to
               increase the punishment for getting a positive wrong that way it will want to put positive more
            3. Many more positive examples. We weight the positive value less so that negatives have a chance.
    '''
    pos = 0.0
    neg = 0.0
    for img_name in tqdm(train_data_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(root, 'masks', img_name[0])))
        for slic in img:
            if not np.any(slic):
                continue
            else:
                p = np.count_nonzero(slic)
                pos += p
                neg += (slic.size - p)

    return neg/pos

def load_class_weights(root, split):
    class_weight_filename = join(root, 'split_lists', 'train_split_' + str(split) + '_class_weights.npy')
    try:
        return np.load(class_weight_filename)
    except:
        print('\nClass weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _, _ = load_data(root, str(split))
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename,value)
        print('\nFinished computing class weights. This value has been saved for this training split.')
        return value


def split_data(root_path, num_splits):
    mask_list = []
    for ext in ('*.mhd', '*.hdr', '*.nii', '*.png', '*.nii.gz'): #add png file support
        mask_list.extend(sorted(glob(join(root_path,'masks',ext)))) # check imgs instead of masks

    assert len(mask_list) != 0, 'Unable to find any files in {}'.format(join(root_path,'masks'))
    print(mask_list)
    outdir = join(root_path,'split_lists')
    try:
        makedirs(outdir)
        print("Made directory")
    except:
        pass
        print("Could not make dir {}".format(outdir))

    if num_splits == 1:
        # Testing model, training set = testing set = 1 image
        train_index = test_index = mask_list
        with open(join(outdir,'train_split_' + str(0) + '.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            print('basename=%s'%([basename(mask_list[0])]))
            writer.writerow([basename(mask_list[0])])
        with open(join(outdir,'test_split_' + str(0) + '.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([basename(mask_list[0])])
     
    else:
        kf = KFold(n_splits=num_splits)
        n = 0
        for train_index, test_index in kf.split(mask_list):
            with open(join(outdir,'train_split_' + str(n) + '.csv'), 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in train_index:
                    writer.writerow([basename(mask_list[i])])
            with open(join(outdir,'test_split_' + str(n) + '.csv'), 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in test_index:
                    writer.writerow([basename(mask_list[i])])
            n += 1
            
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


''' Make the generators threadsafe in case of multiple threads '''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

mean = np.array([18.426106306720985, 24.430354760142666, 24.29803657467962, 19.420110564555472])
std = np.array([104.02684046042094, 136.06477850668273, 137.4833895418739, 109.29833288911334])
one_hot_max = 1.0 # Value of positive class in one hot


def convert_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False):
    fname = img_name[:-7]
    numpy_path = join(root_path, 'np_files')
    img_path = join(root_path, 'imgs')
    mask_path = join(root_path, 'masks')
    fig_path = join(root_path, 'figs')
    try:
        makedirs(numpy_path)
    except:
        pass
    try:
        makedirs(fig_path)
    except:
        pass
    # The min and max pixel values in a ct image file
    brats_min = -0.18
    brats_max = 10

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:
        itk_img = sitk.ReadImage(join(img_path, img_name))

        img = sitk.GetArrayFromImage(itk_img)

        img = img.astype(np.float32)

        img = np.rollaxis(img, 0, 4)
        img = np.rollaxis(img, 0, 3) 
        
      
        img -= mean
        img /= std
        
        img = np.clip(img, + brats_min, brats_max)
        img = (img - brats_min) / (brats_max - brats_min)
        
        #img = img[:, :, :, 3] # Select only t1w during initial testing
        #img = (img-img.mean())/img.std()
        
        
        if not no_masks:
            itk_mask = sitk.ReadImage(join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)
            #mask[mask < 0.5] = 0 # Background
            #mask[mask > 0.5] = 1 # Edema, Enhancing and Non enhancing tumor
            
            label = mask.astype(np.int64)
            masks = np.eye(4)[label]
            print("Created mask shape: {}".format(masks.shape))
            #mask = masks.astype(np.float32)
            
            #masks[masks>0.5] = one_hot_max
            #masks[masks<0.5] = 1-one_hot_max
            mask = masks
            #mask = masks.astype(np.uint8)
            

        try:
            show_modal = 3
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3, show_modal], cmap='gray')
            if not no_masks:
                ax[0].imshow(mask[:, :, img.shape[2] // 3], alpha=0.40, cmap='Reds')
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2, show_modal], cmap='gray')
            if not no_masks:
                ax[1].imshow(mask[:, :, img.shape[2] // 2], alpha=0.40, cmap='Reds')
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4, show_modal], cmap='gray')
            if not no_masks:
                ax[2].imshow(mask[:, :, img.shape[2] // 2 + img.shape[2] // 4], alpha=0.40, cmap='Reds')
            ax[2].set_title('Slice {}/{}'.format(img.shape[2] // 2 + img.shape[2] // 4, img.shape[2]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle(fname)
            print("save qual fig")
            plt.savefig(join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print('\n'+'-'*100)
            print('Error creating qualitative figure for {}'.format(fname))
            print(e)
            print('-'*100+'\n')

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)


@threadsafe_generator
@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    # Create placeholders for training
    # (img_shape[1], img_shape[2], args.slices)
    print(net_input_shape)
    modalities = net_input_shape[2] // numSlices
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    print("img_batch " + str(img_batch.shape))
    mask_shape = [net_input_shape[0],net_input_shape[1], 4]
    mask_batch = np.zeros((np.concatenate(((batchSize,), mask_shape))), dtype=np.float32)
    mask_batch[:, :, :, :] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-6]+'npz')
                print('\npath_to_np=%s'%(path_to_np))
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-7]))
                train_img, train_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
            #print("Train mask shape {}".format(train_mask.shape))

            if numSlices == 1:
                sideSlices = 0
            else:
                if numSlices % 2 != 0:
                    numSlices -= 1
                sideSlices = numSlices / 2

            indicies = np.arange(0, train_img.shape[2], stride)
            z_shape = train_img.shape[2]
            
            if shuff:
                shuffle(indicies)
            for j in indicies:
                if not np.any(train_mask[:, :, j : j+numSlices]):
                    continue
                if aug_data:
                    train_img, train_mask = augment_random(train_img, train_mask)
                if img_batch.ndim == 4:
                    img_batch[count] = 0
                    next_img = train_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(240, 240, -1)
                    insertion_index = -modalities
                    img_index = 0
                    for k in range(j-sideSlices, j+sideSlices+1):
                        insertion_index += modalities
                        if (k < 0): continue
                        if (k >= z_shape): break
                        img_batch[count, 8:-8, 8:-8, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                        img_index += modalities
                    mask_batch[count] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])
                    mask_batch[count, 8:-8, 8:-8, :] = train_mask[:, :, j]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count] = 0
                    mask_batch[count] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])
                    img_batch[count, 8:-8, 8:-8, :, :] = train_img[:, :, j-sideSlices : j+sideSlices+1]
                    mask_batch[count, 8:-8, 8:-8, :, :] = train_mask[:, :, j]
                else:
                    print('\nError this function currently only supports 2D and 3D data.')
                    exit(0)
                    
                if aug_data:
                    img_batch[count], mask_batch[count] = elasticDeform2D(img_batch[count], mask_batch[count], alpha=720, sigma=24, mode='reflect', is_random=True)
                count += 1
                if count % batchSize == 0:
                    count = 0
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_slice1.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()
                            '''plt.imshow(np.squeeze(img_batch[0, :, :, 4]), cmap='gray')
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_slice2.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()
                            plt.imshow(np.squeeze(img_batch[0, :, :, 8]), cmap='gray')
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_slice3_main.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_label.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()
                            plt.imshow(np.squeeze(img_batch[0, :, :, 12]), cmap='gray')
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_slice4.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()
                            plt.imshow(np.squeeze(img_batch[0, :, :, 16]), cmap='gray')
                            plt.savefig(join(root_path, 'logs', 'ex{}_train_slice5.png'.format(j)), format='png', bbox_inches='tight')
                            plt.close()'''
                        '''elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(join(root_path, 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()'''
                    if net.find('caps') != -1: # if the network is capsule/segcaps structure
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1):
    # Create placeholders for validation
    modalities = net_input_shape[2] // numSlices
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_shape = [net_input_shape[0],net_input_shape[1], 4]
    mask_batch = np.zeros((np.concatenate(((batchSize,), mask_shape))), dtype=np.float32)
    mask_batch[:, :, :, :] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan_name in enumerate(val_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-6]+'npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-7]))
                val_img, val_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            if numSlices == 1:
                sideSlices = 0
            else:
                if numSlices % 2 != 0:
                    numSlices -= 1
                sideSlices = numSlices / 2

            indicies = np.arange(0, val_img.shape[2], stride)
            z_shape = val_img.shape[2]
            
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(val_mask[:, :,  j:j+numSlices]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count] = 0
                    next_img = val_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(240, 240, -1)
                    insertion_index = -modalities
                    img_index = 0
                    for k in range(j-sideSlices, j+sideSlices+1):
                        insertion_index += modalities
                        if (k < 0): continue
                        if (k >= z_shape): break
                        img_batch[count, 8:-8, 8:-8, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                        img_index += modalities
                    
                    mask_batch[count] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])
                    mask_batch[count, 8:-8, 8:-8, :] = val_mask[:, :, j]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count] = 0
                    mask_batch[count] = np.array([one_hot_max,1-one_hot_max,1-one_hot_max,1-one_hot_max])
                    img_batch[count, 8:-8, 8:-8, :, :] = val_img[:, :, j : j+numSlices]
                    mask_batch[count, 8:-8, 8:-8, :, :] = val_mask[:, :, j]
                else:
                    print('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    # Create placeholders for testing
    print('\nload_3D_data.generate_test_batches')
    print("Batch size {}".format(batchSize))
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    modalities = net_input_shape[2] // numSlices
    count = 0
    print('\nload_3D_data.generate_test_batches: test_list=%s'%(test_list))
    for i, scan_name in enumerate(test_list):
        try:
            scan_name = scan_name[0]
            path_to_np = join(root_path,'np_files',basename(scan_name)[:-6]+'npz')
            print(path_to_np)
            with np.load(path_to_np) as data:
                test_img = data['img']
        except Exception as err:
            print(err)
            print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-7]))
            test_img = convert_data_to_numpy(root_path, scan_name, no_masks=False)[0]
            if np.array_equal(test_img,np.zeros(1)):
                continue
            else:
                print('\nFinished making npz file.')

        if numSlices == 1:
            sideSlices = 0
        else:
            if numSlices % 2 != 0:
                numSlices -= 1
            sideSlices = numSlices / 2

            indicies = np.arange(0, test_img.shape[2], stride)
            z_shape = test_img.shape[2]

        #print(test_img.shape)
        indicies = np.arange(0, test_img.shape[2], stride)
        for j in indicies:
            if img_batch.ndim == 4:
                img_batch[count] = 0
                next_img = test_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(240, 240, -1)
                insertion_index = -modalities
                img_index = 0
                for k in range(j-sideSlices, j+sideSlices+1):
                    insertion_index += modalities
                    if (k < 0): continue
                    if (k >= z_shape): break
                    img_batch[count, 8:-8, 8:-8, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                    img_index += modalities
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, 8:-8, 8:-8, :, :] = test_img[:, :,  j : j+numSlices]
            else:
                print('Error this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
        
