'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss, multiclass_dice_loss, multiclass_dice_score
from load_brats_data_multiclass import load_class_weights, generate_train_batches, generate_val_batches




def get_loss(root, split, net, recon_wei, choice):
    if choice == 'w_bce':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = weighted_binary_crossentropy_loss(pos_class_weight)
    elif choice == 'bce':
        loss = 'binary_crossentropy'
    elif choice == 'dice':
        loss = dice_loss
    elif choice == 'multi_dice':
        loss = multiclass_dice_loss
    elif choice == 'w_mar':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    elif choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
    else:
        return loss, None

def get_callbacks(arguments):
    if arguments.net.find('caps') != -1:
        if "multi" in arguments.loss:
            monitor_name = 'val_out_seg_multiclass_dice_score'
        else:
            monitor_name = 'val_out_seg_dice_hard'
        
    else:
        if "multi" in arguments.loss:
            monitor_name = 'val_multiclass_dice_score'
        else:
            monitor_name = 'val_dice_hard'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=False,
                                       verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.25, cooldown=0, patience=5,verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, net_input_shape, uncomp_model):
    # Set optimizer loss and metrics
    opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    if args.net.find('caps') != -1:
        if "multi" in args.loss:
            metrics = {'out_seg': multiclass_dice_score}
        else:
            metrics = {'out_seg': dice_hard}
        
    else:
        if "multi" in args.loss:
            metrics = [multiclass_dice_score]
        else:
            metrics = [dice_hard]

    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net,
                                    recon_wei=args.recon_wei, choice=args.loss)

    # If using CPU or single GPU
    if args.gpus <= 1:
        print(loss)
        print(loss_weighting)
        uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return uncomp_model
    # If using multiple GPUs
    else:

        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return model


def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:

        if "multi" in arguments.loss:
            ax1.plot(training_history.history['out_seg_multiclass_dice_score'])
            ax1.plot(training_history.history['val_out_seg_multiclass_dice_score'])
        else:
            ax1.plot(training_history.history['out_seg_dice_hard'])
            ax1.plot(training_history.history['val_out_seg_dice_hard'])
    else:
        if "multi" in arguments.loss:
            ax1.plot(training_history.history['multiclass_dice_score'])
            ax1.plot(training_history.history['val_multiclass_dice_score'])
        else:
            ax1.plot(training_history.history['dice_hard'])
            ax1.plot(training_history.history['val_dice_hard'])
        
    ax1.set_title('Dice Coefficient')
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_dice_hard'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['dice_hard'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()

DEBUG = False
def train(args, train_list, val_list, u_model, net_input_shape, num_output_classes=2):
    # Compile the loaded model
    print(args.dataset)
    print(net_input_shape)
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=u_model)
    if args.weights_path:
        weights_path = join(args.data_root_dir, args.weights_path)
        try:
            model.load_weights(weights_path)
        except:
            assert False, 'Unable to find weights path.'
    # Set the callbacks
    callbacks = get_callbacks(args)
    #val_list = train_list

    
    
    if DEBUG:
        batch_gen = generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                                   batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                                   stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data, dataset=args.dataset, num_output_classes=num_output_classes)
        #import random
        for batch in batch_gen:
            imgs, masks = batch
            single_img = imgs[0]
            single_mask = masks[0]
            print(imgs.shape)
            #if random.randint(0,1) > 0.9:
            break


        f, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].imshow(single_img[:, :, 0], alpha=1, cmap='gray')
        #ax[0].imshow(output_bin[num_slices // 3, :, :], alpha=0.5, cmap='Blues')
        #ax[0].imshow(gt_data[num_slices // 3, :, :], alpha=0.2, cmap='Reds')
        ax[0].set_title('Img')
        ax[0].axis('off')

        ax[1].imshow(single_mask[:, :, 0], alpha=1, cmap='gray')
        #ax[0].imshow(output_bin[num_slices // 3, :, :], alpha=0.5, cmap='Blues')
        #ax[0].imshow(gt_data[num_slices // 3, :, :], alpha=0.2, cmap='Reds')
        ax[1].set_title('Img')
        ax[1].axis('off')

        #ax[1].imshow(img_data[num_slices // 2, :, :], alpha=1, cmap='gray')
        #ax[1].imshow(output_bin[num_slices // 2, :, :], alpha=0.5, cmap='Blues')
        #ax[1].imshow(gt_data[num_slices // 2, :, :], alpha=0.2, cmap='Reds')
        #ax[1].set_title('Mask')
        #ax[1].axis('off')

        fig = plt.gcf()
        fig.suptitle("Input image and mask")

        plt.savefig(join("figz/", 'img_mask_fig' + '.png'),
                    format='png', bbox_inches='tight')
        plt.close('all') 
        
        for line in range(single_mask.shape[0]):
            for col in range(single_mask.shape[1]):
                print(single_mask[line,col], end=" ")
            print("")
        print(single_mask.shape)
        #print(single_mask)

        assert False

    # Training the network
    history = model.fit_generator(
        generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                               stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data, dataset=args.dataset, 
                               num_output_classes=num_output_classes),
        max_queue_size=args.max_queue_size, workers=args.workers, use_multiprocessing=args.use_multiprocessing==1,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=generate_val_batches(args.data_root_dir, val_list, net_input_shape, net=args.net,
                                             batchSize=args.batch_size,  numSlices=args.slices, subSampAmt=0,
                                             stride=5, shuff=args.shuffle_data, dataset=args.dataset, 
                                             num_output_classes=num_output_classes),
        validation_steps=args.validation_steps, # Set validation stride larger to see more of the data.
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1)

    # Plot the training data collected
    #plot_training(history, args)
