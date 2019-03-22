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
from metrics import dc, jc, assd, jaccard

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils import print_summary
from msd_metrics import compute_dice_coefficient, compute_surface_dice_at_tolerance, compute_surface_distances, compute_surface_dice_at_tolerance


from load_data_multiclass import generate_test_batches
from postprocess import oneHot2LabelMin, oneHot2LabelMax

def create_activation_image(args, raw_data, label, slice_num = 77, index=0):
    if raw_data.shape[3] == 1:
        f, ax = plt.subplots(2, raw_data.shape[3]+1, figsize=(15, 15))
    else:
        f, ax = plt.subplots(2, raw_data.shape[3], figsize=(15, 15))

    print(raw_data.shape)
    names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    mi = np.min(raw_data[slice_num])
    ma = np.max(raw_data[slice_num])

    for j in range(raw_data.shape[3]):
        ax[0,j].imshow(raw_data[slice_num, :, :, j], alpha=1, cmap='Reds', vmin=mi, vmax=ma)
        ax[0,j].set_title(names[j])
        ax[0,j].axis('off')

    for j in range(raw_data.shape[3]):
        ax[1,j].imshow(label[slice_num, :, :, j], alpha=1, cmap='Reds')
        ax[1,j].set_title(names[j])
        ax[1,j].axis('off')


    fig = plt.gcf()
    fig.suptitle("Activation maps for img {}".format(index))

    plt.savefig(join(args.data_root_dir, 'results', 'activations', 'img_{}'.format(index) + '.png'),
                format='png', bbox_inches='tight')
    plt.close('all')

def create_recon_image(args, recon, img, i=0):
    cols = recon.shape[3]
    if cols == 1:
        cols = 2
    f, ax = plt.subplots(2, cols, figsize=(15, 15))

    slice_num = recon.shape[0] // 2
    for j in range(recon.shape[3]):
        recon_mod = recon[slice_num, :, :, j]
        ax[0,j].imshow(recon_mod, alpha=1, cmap='Reds')
        ax[0,j].axis('off')

    print('recon shape' + str(img.shape))
    for j in range(recon.shape[3]):
        ax[1,j].imshow(img[slice_num, :, :], alpha=1, cmap='Reds')
        ax[1,j].axis('off')

    fig = plt.gcf()
    fig.suptitle("Recon maps for img {}".format(i))

    plt.savefig(join(args.data_root_dir, 'results', 'activations', 'recon_{}'.format(i) + '.png'),
                format='png', bbox_inches='tight')
    plt.close('all')

def threshold_mask(raw_output, threshold):

    raw_output[raw_output > 0.5] = 1
    raw_output[raw_output < 0.5] = 0


    return raw_output

def create_class_mask(pred, gt, class_idx):
    pred_class = np.full(pred.shape, False, dtype=bool)
    gt_class = np.full(gt.shape, False, dtype=bool)
    pred_class[np.where(pred == class_idx)] = True
    gt_class[np.where(gt == class_idx)] = True
    
    return pred_class, gt_class
    

def calc_dice_scores(pred, gt, num_classes):
    scores = np.zeros(num_classes - 1)
    for class_idx in range(1, num_classes):
        pred_class, gt_class = create_class_mask(pred, gt, class_idx)
        coeff = compute_dice_coefficient(gt_class, pred_class)
        other_dice = dc(pred_class, gt_class)
        #print(other_dice)
        print("Dice value: {}".format(coeff))
        scores[class_idx-1] = coeff
    return scores

def calc_assd_scores(pred, gt, num_classes, spacing):
    scores = np.zeros(num_classes - 1)
    for class_idx in range(1, num_classes):
        pred_class, gt_class = create_class_mask(pred, gt, class_idx)
        score_dict = compute_surface_distances(gt_class, pred_class, spacing)
        score = compute_surface_dice_at_tolerance(score_dict, 4)
        assd_score = assd(pred_class, gt_class, voxelspacing=spacing, connectivity=1)
        
        print(score)
        print(assd_score)
        #other_dice = dc(pred_class, gt_class)
        #print(other_dice)
        #print(coeff)
        #scores[class_idx-1] = score
    return scores


def test(args, test_list, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_validation_best_model_' + args.time + '.hdf5')
    else:
        weights_path = join(args.data_root_dir, args.weights_path)

    if args.dataset == 'brats':
        RESOLUTION = 240
    elif args.dataset == 'heart':
        RESOLUTION = 320
    else:
        RESOLUTION = 512

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

    # Set up placeholders
    outfile = ''
    if args.compute_dice:
        dice_arr = np.zeros((len(test_list)))
        outfile += 'dice_'
    if args.compute_jaccard:
        jacc_arr = np.zeros((len(test_list)))
        outfile += 'jacc_'
    if args.compute_assd:
        assd_arr = np.zeros((len(test_list)))
        outfile += 'assd_'
    surf_arr = np.zeros((len(test_list)), dtype=str)
    dice2_arr = np.zeros((len(test_list), args.out_classes-1))

    # Testing the network
    print('Testing... This will take some time...')

    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name']
        if args.compute_dice:
            row.append('Dice Coefficient')
        if args.compute_jaccard:
            row.append('Jaccard Index')
        if args.compute_assd:
            row.append('Average Symmetric Surface Distance')

        writer.writerow(row)

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

            print('out' + str(output_array[0].shape))
            if args.net.find('caps') != -1:
                if args.dataset == 'brats':
                    output = output_array[0][:,8:-8,8:-8]
                    recon = output_array[1][:,8:-8,8:-8]
                else:
                    output = output_array[0][:,:,:]
                    recon = output_array[1][:,:,:]
            else:
                if args.dataset == 'brats':
                    output = output_array[:,8:-8,8:-8,:]
                else:
                    output = output_array[:,:,:,:]


            if args.out_classes == 1:
                output_raw = output.reshape(-1,RESOLUTION,RESOLUTION,1) #binary
            else:
                output_raw = output
                output = oneHot2LabelMax(output)

            out = output.astype(np.int64)

            if args.out_classes == 1:
                outputOnehot = out.reshape(-1,RESOLUTION,RESOLUTION,1) #binary
            else:
                outputOnehot = np.eye(args.out_classes)[out].astype(np.uint8)


            output_img = sitk.GetImageFromArray(output)

            print('Segmenting Output')

            output_bin = threshold_mask(output, args.thresh_level)

            output_mask = sitk.GetImageFromArray(output_bin)

            slice_img = sitk.Image(RESOLUTION,RESOLUTION,num_slices, sitk.sitkUInt8)

            output_img.CopyInformation(slice_img)
            output_mask.CopyInformation(slice_img)

            if args.net.find('caps') != -1:
                create_recon_image(args, recon, img_data, i=i)


            #output_img.CopyInformation(sitk_img)
            #output_mask.CopyInformation(sitk_img)

            print('Saving Output')
            if args.dataset != 'luna':
                sitk.WriteImage(output_img, join(raw_out_dir, img[0][:-7] + '_raw_output' + img[0][-7:]))
                sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-7] + '_final_output' + img[0][-7:]))

                # Load gt mask
                sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))
                gt_data = sitk.GetArrayFromImage(sitk_mask)
                label = gt_data.astype(np.int64)

                if args.out_classes == 1:
                    gtOnehot = label.reshape(-1,RESOLUTION,RESOLUTION,1) #binary
                    gt_label = label
                else:
                    gtOnehot = np.eye(args.out_classes)[label].astype(np.uint8)
                    gt_label = label

                create_activation_image(args, output_raw, gtOnehot, slice_num=output_raw.shape[0] // 2, index=i)
                # Plot Qual Figure
                print('Creating Qualitative Figure for Quick Reference')
                f, ax = plt.subplots(2, 3, figsize=(10, 5))

                colors = ['Greys', 'Greens', 'Reds', 'Blues']
                fileTypeLength = 7

                print(img_data.shape)
                print(outputOnehot.shape)
                if args.dataset == 'brats':
                    img_data = img_data[3]
                    #img_data = img_data[:,:,:,3]

                ax[0,0].imshow(img_data[num_slices // 3, :, :], alpha=1, cmap='gray')

                for class_num in range(outputOnehot.shape[3]):
                    ax[0,0].imshow(outputOnehot[num_slices // 3, :, :, class_num], alpha=0.5, cmap=colors[class_num], vmin = 0, vmax = 1)

                ax[0,0].set_title('Slice {}/{}'.format(num_slices // 3, num_slices))
                ax[0,0].axis('off')

                ax[0,1].imshow(img_data[num_slices // 2, :, :], alpha=1, cmap='gray')
                for class_num in range(outputOnehot.shape[3]):
                    ax[0,1].imshow(outputOnehot[num_slices // 2, :, :, class_num], alpha=0.5, cmap=colors[class_num], vmin = 0, vmax = 1)
                ax[0,1].set_title('Slice {}/{}'.format(num_slices // 2, num_slices))
                ax[0,1].axis('off')

                ax[0,2].imshow(img_data[num_slices // 2 + num_slices // 4, :, :], alpha=1, cmap='gray')
                for class_num in range(outputOnehot.shape[3]):
                    ax[0,2].imshow(outputOnehot[num_slices // 2 + num_slices // 4, :, :, class_num], alpha=0.5, cmap=colors[class_num], vmin = 0, vmax = 1)
                #ax[0,2].imshow(gt_data[num_slices // 2 + num_slices // 4, :, :], alpha=0.2,cmap='Reds')
                ax[0,2].set_title(
                    'Slice {}/{}'.format(num_slices // 2 + num_slices // 4, num_slices))
                ax[0,2].axis('off')


                #print(gt_data[num_slices // 3, :, :])
                ax[1,0].imshow(img_data[num_slices // 3, :, :], alpha=1, cmap='gray')
                ax[1,0].set_title('Slice {}/{}'.format(num_slices // 3, num_slices))
                ax[1,0].imshow(gt_data[num_slices // 3, :, :], alpha=0.8, cmap='Reds', vmin=0, vmax=args.out_classes-1)
                ax[1,0].axis('off')

                ax[1,1].imshow(img_data[num_slices // 2, :, :], alpha=1, cmap='gray')
                ax[1,1].set_title('Slice {}/{}'.format(num_slices // 2, num_slices))
                ax[1,1].imshow(gt_data[num_slices // 2, :, :], alpha=0.8, cmap='Reds', vmin=0, vmax=args.out_classes-1)
                ax[1,1].axis('off')

                ax[1,2].imshow(img_data[num_slices // 2 + num_slices // 4, :, :], alpha=1, cmap='gray')
                ax[1,2].imshow(gt_data[num_slices // 2 + num_slices // 4, :, :], alpha=0.8, cmap='Reds', vmin=0, vmax=args.out_classes-1)
                ax[1,2].set_title(
                    'Slice {}/{}'.format(num_slices // 2 + num_slices // 4, num_slices))
                ax[1,2].axis('off')

                fig = plt.gcf()
                fig.suptitle(img[0][:-fileTypeLength])

                plt.savefig(join(fig_out_dir, img[0][:-fileTypeLength] + '_qual_fig' + '.png'),
                            format='png', bbox_inches='tight')
                plt.close('all')
            else:
                sitk.WriteImage(output_img, join(raw_out_dir, img[0][:-4] + '_raw_output' + img[0][-4:]))
                sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]))

                # Load gt mask
                sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))
                gt_data = sitk.GetArrayFromImage(sitk_mask)

                f, ax = plt.subplots(1, 3, figsize=(15, 5))

                ax[0].imshow(img_data[img_data.shape[0] // 3, :, :], alpha=1, cmap='gray')
                ax[0].imshow(output_bin[img_data.shape[0] // 3, :, :], alpha=0.5, cmap='Reds')
                #ax[0].imshow(gt_data[img_data.shape[0] // 3, :, :], alpha=0.2, cmap='Reds')
                ax[0].set_title('Slice {}/{}'.format(img_data.shape[0] // 3, img_data.shape[0]))
                ax[0].axis('off')

                ax[1].imshow(img_data[img_data.shape[0] // 2, :, :], alpha=1, cmap='gray')
                ax[1].imshow(output_bin[img_data.shape[0] // 2, :, :], alpha=0.5, cmap='Reds')
                #ax[1].imshow(gt_data[img_data.shape[0] // 2, :, :], alpha=0.2, cmap='Reds')
                ax[1].set_title('Slice {}/{}'.format(img_data.shape[0] // 2, img_data.shape[0]))
                ax[1].axis('off')

                ax[2].imshow(img_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=1, cmap='gray')
                ax[2].imshow(output_bin[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.5,
                             cmap='Reds')
                #ax[2].imshow(gt_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.2,
                #             cmap='Reds')
                ax[2].set_title(
                    'Slice {}/{}'.format(img_data.shape[0] // 2 + img_data.shape[0] // 4, img_data.shape[0]))
                ax[2].axis('off')

                fig = plt.gcf()
                fig.suptitle(img[0][:-4])

                plt.savefig(join(fig_out_dir, img[0][:-4] + '_qual_fig' + '.png'),
                            format='png', bbox_inches='tight')

                
            output_label = oneHot2LabelMax(outputOnehot)

            row = [img[0][:-4]]
            if args.compute_dice:
                print('Computing Dice')
                dice_arr[i] = dc(outputOnehot, gtOnehot)
                print('\tDice: {}'.format(dice_arr[i]))
                row.append(dice_arr[i])
            if args.compute_jaccard:
                print('Computing Jaccard')
                jacc_arr[i] = jaccard(outputOnehot, gtOnehot)
                print('\tJaccard: {}'.format(jacc_arr[i]))
                row.append(jacc_arr[i])
            if args.compute_assd:
                print('Computing ASSD')
                assd_arr[i] = assd(outputOnehot, gtOnehot, voxelspacing=sitk_img.GetSpacing(), connectivity=1)
                print('\tASSD: {}'.format(assd_arr[i]))
                row.append(assd_arr[i])
            try:
                spacing = np.array(sitk_img.GetSpacing())
                if args.dataset == 'brats':
                   spacing = spacing[1:]
                surf = compute_surface_distances(label, out, spacing)
                surf_arr[i] = str(surf)
                assd_score = calc_assd_scores(output_label, gt_label, args.out_classes, spacing)
                print(assd_score)
                print('\tSurface distance ' + str(surf_arr[i]))
            except:
                print("surf failed")
                pass
            #dice2_arr[i] = compute_dice_coefficient(gtOnehot, outputOnehot)
            dice2_arr[i] = calc_dice_scores(output_label, gt_label, args.out_classes)
            
            print('\tMSD Dice: {}'.format(dice2_arr[i]))
            
            writer.writerow(row)

        row = ['Average Scores']
        if args.compute_dice:
            row.append(np.mean(dice_arr))
        if args.compute_jaccard:
            row.append(np.mean(jacc_arr))
        if args.compute_assd:
            row.append(np.mean(assd_arr))
        row.append(surf_arr)
        row.append(np.mean(dice2_arr))
        
        writer.writerow(row)

    print('Done.')
