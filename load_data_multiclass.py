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
from load_heart import convert_heart_data_to_numpy
from load_spleen import convert_spleen_data_to_numpy
from load_brats import convert_brats_data_to_numpy
from postprocess import oneHot2LabelMax
from augmentation import augment_random, elasticDeform2D, elasticDeform3D

from scipy import linalg


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

one_hot_max = 1.0 # Value of positive class in one hot


@threadsafe_generator
@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1, dataset = 'brats', num_output_classes=2):
    # Create placeholders for training
    # (img_shape[1], img_shape[2], args.slices)
    print('train ' + str(dataset))
    modalities = net_input_shape[2] // numSlices
    input_slices = numSlices
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_shape = [net_input_shape[0], net_input_shape[1], num_output_classes]
    #print(mask_shape)
    mask_batch = np.zeros((np.concatenate(((batchSize,), mask_shape))), dtype=np.float32)

    if dataset == 'brats':
        np_converter = convert_brats_data_to_numpy
        frame_pixels_0 = 8
        frame_pixels_1 = -8
        empty_mask = np.array([one_hot_max, 1-one_hot_max, 1-one_hot_max, 1-one_hot_max])
        raw_x_shape = 240
        raw_y_shape = 240
    elif dataset in ['heart', 'spleen']:
        if dataset == 'heart':
            np_converter = convert_heart_data_to_numpy
        else:
            np_converter = convert_spleen_data_to_numpy
        frame_pixels_0 = 0
        frame_pixels_1 = net_input_shape[0]
        if num_output_classes == 2:
            empty_mask = np.array([one_hot_max, 1-one_hot_max])
        else:
            empty_mask = np.array([1-one_hot_max])
        raw_x_shape = net_input_shape[0]
        raw_y_shape = net_input_shape[1]
    else:
        assert False, 'Dataset not recognized'

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        is_binary_classification = num_output_classes == 1
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-6]+'npz')
                #print('\npath_to_np=%s'%(path_to_np))
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                #print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-7]))
                train_img, train_mask = np_converter(root_path, scan_name, num_classes=num_output_classes)
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

            z_shape = train_img.shape[2]
            indicies = np.arange(0, z_shape, stride)

            if shuff:
                shuffle(indicies)
            for j in indicies:

                if (is_binary_classification and np.sum(train_mask[:, :, j]) < 1) or (not is_binary_classification and np.sum(train_mask[:, :, j, 1:]) < 1):
                    #print('hola')
                    continue
                if aug_data:
                    train_img, train_mask = augment_random(train_img, train_mask)
                if img_batch.ndim == 4:
                    img_batch[count] = 0
                    next_img = train_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(raw_x_shape, raw_y_shape, -1)
                    insertion_index = -modalities
                    img_index = 0
                    for k in range(j-sideSlices, j+sideSlices+1):
                        insertion_index += modalities
                        if (k < 0): continue
                        if (k >= z_shape): break
                        img_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                        img_index += modalities
                    mask_batch[count] = empty_mask
                    mask_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, :] = train_mask[:, :, j]
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
                        mid_slice = input_slices // 2
                        start_index = mid_slice * modalities
                        img_batch_mid_slice = img_batch[:, :, :, start_index:start_index+modalities]

                        mask_batch_masked = oneHot2LabelMax(mask_batch)
                        mask_batch_masked[mask_batch_masked > 0.5] = 1.0 # Setting all other classes than background to mask
                        mask_batch_masked = np.expand_dims(mask_batch_masked, axis=-1)
                        mask_batch_masked_expand = np.repeat(mask_batch_masked, modalities, axis=-1)

                        masked_img = mask_batch_masked_expand*img_batch_mid_slice

                        '''plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                        plt.savefig(join(root_path, 'logs', '{}_img.png'.format(j)), format='png', bbox_inches='tight')
                        plt.close()
                        plt.imshow(np.squeeze(mask_batch_masked[0, :, :, 0]), cmap='gray')
                        plt.savefig(join(root_path, 'logs', '{}_mask_masked.png'.format(j)), format='png', bbox_inches='tight')
                        plt.close()
                        plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), cmap='gray')
                        plt.savefig(join(root_path, 'logs', '{}_mask.png'.format(j)), format='png', bbox_inches='tight')
                        plt.close()
                        plt.imshow(np.squeeze(masked_img[0, :, :, 0]), cmap='gray')
                        plt.savefig(join(root_path, 'logs', '{}_masked_img.png'.format(j)), format='png', bbox_inches='tight')
                        plt.close()'''
                        yield ([img_batch, mask_batch_masked], [mask_batch, masked_img])
                    else:
                        yield (img_batch, mask_batch)
        if count != 0:
            #if aug_data:
            #    img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
            #                                                                  mask_batch[:count,...])
            if net.find('caps') != -1:
                mid_slice = input_slices // 2
                start_index = mid_slice * modalities
                img_batch_mid_slice = img_batch[:, :, :, start_index:start_index+modalities]

                mask_batch_masked = oneHot2LabelMax(mask_batch)
                mask_batch_masked[mask_batch_masked > 0.5] = 1.0 # Setting all other classes than background to mask
                mask_batch_masked = np.expand_dims(mask_batch_masked, axis=-1)
                mask_batch_masked_expand = np.repeat(mask_batch_masked, modalities, axis=-1)
                yield ([img_batch[:count, ...], 1 - mask_batch_masked[:count, ...]],
                       [mask_batch[:count, ...], mask_batch_masked_expand[:count, ...] * img_batch_mid_slice[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1, dataset = 'brats', num_output_classes=2):
    # Create placeholders for validation

    modalities = net_input_shape[2] // numSlices
    input_slices = numSlices
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_shape = [net_input_shape[0],net_input_shape[1], num_output_classes]
    mask_batch = np.zeros((np.concatenate(((batchSize,), mask_shape))), dtype=np.float32)

    if dataset == 'brats':
        np_converter = convert_brats_data_to_numpy
        frame_pixels_0 = 8
        frame_pixels_1 = -8
        empty_mask = np.array([one_hot_max, 1-one_hot_max, 1-one_hot_max, 1-one_hot_max])
        raw_x_shape = 240
        raw_y_shape = 240
    elif dataset in ['heart', 'spleen']:
        if dataset == 'heart':
            np_converter = convert_heart_data_to_numpy
        else:
            np_converter = convert_spleen_data_to_numpy
        frame_pixels_0 = 0
        frame_pixels_1 = net_input_shape[0]
        if num_output_classes == 2:
            empty_mask = np.array([one_hot_max, 1-one_hot_max])
        else:
            empty_mask = np.array([1-one_hot_max])
        raw_x_shape = net_input_shape[0]
        raw_y_shape = net_input_shape[1]
    else:
        assert False, 'Dataset not recognized'

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
                val_img, val_mask = np_converter(root_path, scan_name, num_classes=num_output_classes)
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

            z_shape = val_img.shape[2]
            indicies = np.arange(0, z_shape, stride)

            if shuff:
                shuffle(indicies)

            for j in indicies:
                #if not np.any(val_mask[:, :,  j:j+numSlices]):
                #    continue
                if img_batch.ndim == 4:
                    img_batch[count] = 0
                    next_img = val_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(raw_x_shape, raw_y_shape, -1)
                    insertion_index = -modalities
                    img_index = 0
                    for k in range(j-sideSlices, j+sideSlices+1):
                        insertion_index += modalities
                        if (k < 0): continue
                        if (k >= z_shape): break
                        img_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                        img_index += modalities

                    mask_batch[count] = empty_mask
                    mask_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, :] = val_mask[:, :, j]
                else:
                    print('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1: # if the network is capsule/segcaps structure
                        mid_slice = input_slices // 2
                        start_index = mid_slice * modalities
                        img_batch_mid_slice = img_batch[:, :, :, start_index:start_index+modalities]

                        mask_batch_masked = oneHot2LabelMax(mask_batch)
                        mask_batch_masked[mask_batch_masked > 0.5] = 1.0 # Setting all other classes than background to mask
                        mask_batch_masked = np.expand_dims(mask_batch_masked, axis=-1)
                        mask_batch_masked_expand = np.repeat(mask_batch_masked, modalities, axis=-1)

                        masked_img = mask_batch_masked_expand*img_batch_mid_slice
                        yield ([img_batch, 1 - mask_batch_masked], [mask_batch, masked_img])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            #if aug_data:
            #    img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
            #                                                                  mask_batch[:count,...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1, dataset = 'brats', num_output_classes=2):
    # Create placeholders for testing
    print('Generate test batches for ' + str(dataset))
    print('\nload_3D_data.generate_test_batches')
    print("Batch size {}".format(batchSize))
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    modalities = net_input_shape[2] // numSlices
    count = 0
    print('\nload_3D_data.generate_test_batches: test_list=%s'%(test_list))

    if dataset == 'brats':
        np_converter = convert_brats_data_to_numpy
        frame_pixels_0 = 8
        frame_pixels_1 = -8
        raw_x_shape = 240
        raw_y_shape = 240
    elif dataset in ['heart', 'spleen']:
        if dataset == 'heart':
            np_converter = convert_heart_data_to_numpy
        else:
            np_converter = convert_spleen_data_to_numpy
        frame_pixels_0 = 0
        frame_pixels_1 = net_input_shape[0]
        raw_x_shape = net_input_shape[0]
        raw_y_shape = net_input_shape[1]
    else:
        assert False, 'Dataset not recognized'

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
            test_img = np_converter(root_path, scan_name, no_masks=False, num_classes=num_output_classes)[0]
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

        z_shape = test_img.shape[2]
        indicies = np.arange(0, z_shape, stride)

        for j in indicies:
            if img_batch.ndim == 4:
                img_batch[count] = 0
                next_img = test_img[:, :, max(j-sideSlices,0):min(j+sideSlices+1,z_shape)].reshape(raw_x_shape, raw_y_shape, -1)
                insertion_index = -modalities
                img_index = 0
                for k in range(j-sideSlices, j+sideSlices+1):
                    insertion_index += modalities
                    if (k < 0): continue
                    if (k >= z_shape): break
                    img_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, insertion_index:insertion_index+modalities] = next_img[:, :, img_index:img_index+modalities]
                    img_index += modalities
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, frame_pixels_0:frame_pixels_1, frame_pixels_0:frame_pixels_1, :, :] = test_img[:, :,  j : j+numSlices]
            else:
                print('Error this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
