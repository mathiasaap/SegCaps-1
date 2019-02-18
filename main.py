'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test, and manipulate the SegCaps of models.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function

from os.path import join
from os import makedirs
from os import environ
import argparse
import SimpleITK as sitk
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

from keras.utils import print_summary

from load_brats_data_multiclass import load_data, split_data
from model_helper import create_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def main(args):
    # Ensure training, testing, and manip are not all turned off
    assert (args.train or args.test or args.manip), 'Cannot have train, test, and manip all set to 0, Nothing to do.'

    # Load the training, validation, and testing data
    try:
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)
    except:
        # Create the training and test splits if not found
        split_data(args.data_root_dir, num_splits=args.num_splits)
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)

    # Get image properties from first image. Assume they are all the same.
    print(train_list)
    img_shape = sitk.GetArrayFromImage(sitk.ReadImage(join(args.data_root_dir, 'imgs', train_list[0][0]))).shape
    print(img_shape)
    
    modalities = 1
    if args.dataset == 'brats':
        modalities = 4
    net_input_shape = (img_shape[1], img_shape[2], args.slices*modalities)

    # Create the model for training/testing/manipulation
    model_list = create_model(args=args, input_shape=net_input_shape)
    print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

    args.output_name = 'split-' + str(args.split_num) + '_batch-' + str(args.batch_size) + \
                       '_shuff-' + str(args.shuffle_data) + '_aug-' + str(args.aug_data) + \
                       '_loss-' + str(args.loss) + '_slic-' + str(args.slices) + \
                       '_sub-' + str(args.subsamp) + '_strid-' + str(args.stride) + \
                       '_lr-' + str(args.initial_lr) + '_recon-' + str(args.recon_wei)
    args.time = time

    args.check_dir = join(args.data_root_dir,'saved_models', args.net)
    try:
        makedirs(args.check_dir)
    except:
        pass

    args.log_dir = join(args.data_root_dir,'logs', args.net)
    try:
        makedirs(args.log_dir)
    except:
        pass

    args.tf_log_dir = join(args.log_dir, 'tf_logs')
    try:
        makedirs(args.tf_log_dir)
    except:
        pass

    args.output_dir = join(args.data_root_dir, 'plots', args.net)
    try:
        makedirs(args.output_dir)
    except:
        pass

    if args.train:
        from train import train
        # Run training
        train(args, train_list, val_list, model_list[0], net_input_shape, num_output_classes=args.out_classes)
        args.weights_path = ''

    if args.test:
        if args.dataset == 'brats' or args.dataset == 'heart':
            from test_multiclass import test
            test(args, test_list, model_list, net_input_shape)
        else:
            from test import test
            # Run testing
            test(args, test_list, model_list, net_input_shape)

    if args.manip:
        from manip import manip
        # Run manipulation of segcaps
        manip(args, test_list, model_list, net_input_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    parser.add_argument('--num_splits', type=int, default=2,
                        help='Number of training splits to create.')
    parser.add_argument('--net', type=str.lower, default='segcapsr3',
                        choices=['segcapsr3', 'segcapsr1', 'segcapsbasic', 'unet', 'tiramisu', 'isensee'],
                        help='Choose your network.')
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--manip', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable manipulation.')
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to use data augmentation during training.')
    parser.add_argument('--loss', type=str.lower, default='multi_dice', choices=['bce', 'w_bce', 'dice', 'mar', 'w_mar', 'multi_dice'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--recon_wei', type=float, default=131.072,
                        help="If using capsnet: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--slices', type=int, default=1,
                        help='Number of slices to include for training/testing.')
    parser.add_argument('--dataset', type=str.lower, default='brats', choices=['brats', 'luna16', 'heart'],
                        help='Which dataset to use.')
    parser.add_argument('--out_classes', type=int, default=4,
                        help='Number of classes used by dataset.')
    
    parser.add_argument('--subsamp', type=int, default=-1,
                        help='Number of slices to skip when forming 3D samples for training. Enter -1 for random '
                             'subsampling up to 5% of total slices.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move when generating the next sample.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--save_raw', type=int, default=1, choices=[0,1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_seg', type=int, default=1, choices=[0,1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for otsu thresholding, else set value')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to run. Any positive integer')
    parser.add_argument('--out_classes', type=int, default=2,
                        help='Number of output classes. Any positive integer')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of training steps to run every epoch. Any positive integer')
    parser.add_argument('--validation_steps', type=int, default=600,
                        help='Number of validation steps to run every epoch. Any positive integer')
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')

    parser.add_argument('--use_multiprocessing', type=int, default=1,
                help='Use multiprocessing: [0, 1]. Default: 1')
    
    parser.add_argument('--workers', type=int, default=12,
            help='Thread workers')
    
    parser.add_argument('--max_queue_size', type=int, default=32,
        help='Max queue size. Too high will crash due to memory allocation failure.')
    
    
    

    arguments = parser.parse_args()

    #
    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    main(arguments)
