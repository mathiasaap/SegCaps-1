'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
from os import makedirs
import csv
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
import math

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import print_summary

from load_data_multiclass import generate_test_batches
from postprocess import oneHot2LabelMin, oneHot2LabelMax


def test(args, test_list, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_validation_best_model_' + args.time + '.hdf5')
    else:
        weights_path = join(args.data_root_dir, args.weights_path)

    if args.dataset == 'brats':
        RESOLUTION = 240
    elif args.dataset == 'heart':
        RESOLUTION = 320
    elif args.dataset == 'hippocampus':
        RESOLUTION = 32
    else:
        RESOLUTION = 512
    
    RESOLUTION_X = RESOLUTION_Y = RESOLUTION

    output_dir = join(args.data_root_dir, 'results', args.net, 'split_' + str(args.split_num))
    raw_out_dir = join(output_dir, 'raw_output')
    fin_out_dir = join(output_dir, 'final_output')
    fig_out_dir = join(output_dir, 'qual_figs')
    try:
        makedirs(raw_out_dir)
    except:
        pass
    try:
        makedirs(fin_out_dir)
    except:
        pass
    try:
        makedirs(fig_out_dir)
    except:
        pass

    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]
    try:
        eval_model.load_weights(weights_path)
    except Exception as e:
        print(e)
        assert False, 'Unable to find weights path. Testing with random weights.'
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

 
    # Testing the network
    print('Testing... This will take some time...')
    

    for i, img in enumerate(tqdm(test_list)):
        sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))
        img_data = sitk.GetArrayFromImage(sitk_img)
        num_slices = img_data.shape[0]

        if args.dataset == 'brats':
            num_slices = img_data.shape[1]#brats
            img_data = np.rollaxis(img_data,0,4)

        print(args.dataset)

        output_array = eval_model.predict_generator(generate_test_batches(args.data_root_dir, [img],
                                                                          net_input_shape,
                                                                          batchSize=1,
                                                                          numSlices=args.slices,
                                                                          subSampAmt=0,
                                                                          stride=1, dataset = args.dataset, num_output_classes=args.out_classes),
                                                    steps=num_slices, max_queue_size=1, workers=1,
                                                    use_multiprocessing=False, verbose=1)

        raw_x_shape = img_data.shape[1]
        raw_y_shape = img_data.shape[2]
        print(img_data.shape)
        RESOLUTION_Y,RESOLUTION_X = raw_x_shape, raw_y_shape
        frame_pixels_0 = int(math.floor((48.0 - raw_x_shape)/2))
        frame_pixels_1 = 48-int(math.ceil((48.0 - raw_x_shape)/2))

        frame_pixels_0_2 = int(math.floor((48.0 - raw_y_shape)/2))
        frame_pixels_1_2 = 48-int(math.ceil((48.0 - raw_y_shape)/2))

        print('out' + str(output_array[0].shape))
        if args.net.find('caps') != -1:
            if args.dataset == 'brats':
                output = output_array[0][:,8:-8,8:-8]
                recon = output_array[1][:,8:-8,8:-8]
            elif args.dataset == 'hippocampus':
                output = output_array[0][:,frame_pixels_0:frame_pixels_1,frame_pixels_0_2:frame_pixels_1_2]
                recon = output_array[1][:,frame_pixels_0:frame_pixels_1,frame_pixels_0_2:frame_pixels_1_2]
            else:
                output = output_array[0][:,:,:]
                recon = output_array[1][:,:,:]
        else:
            if args.dataset == 'brats':
                output = output_array[:,8:-8,8:-8,:]
            elif args.dataset == 'hippocampus':
                print(output_array.shape)
                output = output_array[:,frame_pixels_0:frame_pixels_1,frame_pixels_0_2:frame_pixels_1_2]
            else:
                output = output_array[:,:,:,:]

        print(output.shape)
        print(img_data.shape)


        if args.out_classes == 1:
            output_raw = output.reshape(-1,RESOLUTION_X,RESOLUTION_Y,1) #binary
        else:
            output_raw = output
            output = oneHot2LabelMax(output)

        out = output.astype(np.int64)

        if args.out_classes == 1:
            outputOnehot = out.reshape(-1,RESOLUTION_X,RESOLUTION_Y,1) #binary
        else:
            if args.dataset == 'hippocampus':
                out[np.where(out>(args.out_classes-1)) ] = 1
            outputOnehot = np.eye(args.out_classes)[out].astype(np.uint8)


        output_img = sitk.GetImageFromArray(output)

        print('Segmenting Output')

        output_bin = output.astype(np.uint8)

        output_mask = sitk.GetImageFromArray(output_bin)

        output_img.CopyInformation(sitk_img)
        output_mask.CopyInformation(sitk_img)


        print('Saving Output')
        if args.dataset != 'luna':
            sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-7] + '_final_output' + img[0][-7:]))
        else:
            sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]))



    print('Done.')
