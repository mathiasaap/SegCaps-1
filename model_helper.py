'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
'''
import tensorflow as tf

def create_model(args, input_shape):
    # If using CPU or single GPU
    num_classes = args.out_classes
    if not num_classes:
        num_classes = 2
    if args.gpus <= 1:
        if args.net == 'unet':
            from unet import UNet
            model = UNet(input_shape, num_classes)
            return [model]
        elif args.net == 'tiramisu':
            from densenets import DenseNetFCN
            model = DenseNetFCN(input_shape)
            return [model]
        elif args.net == 'segcapsr1':
            from capsnet import CapsNetR1
            model_list = CapsNetR1(input_shape)
            return model_list
        elif args.net == 'segcapsr3':
            from capsnet import CapsNetR3
            model_list = CapsNetR3(input_shape, args.modalities, num_classes)
            return model_list
        elif args.net == 'emseg':
            from emsegcaps import SegCapsEM
            model_list = SegCapsEM(input_shape, args.modalities, num_classes)
            return model_list
        elif args.net == 'capsbasic':
            from capsnet import CapsNetBasic
            model_list = CapsNetBasic(input_shape)
            return model_list
        elif args.net == 'isensee':
            from isensee import ResidualUnet2D
            model = ResidualUnet2D(input_shape, args.out_classes)
            return [model]
        elif args.net == 'binarycaps':
            from capsnet import BinaryCapsNetR3
            model_list = BinaryCapsNetR3(input_shape, args.out_classes)
            return model_list
        else:
            raise Exception('Unknown network type specified: {}'.format(args.net))
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            if args.net == 'unet':
                from unet import UNet
                model = UNet(input_shape)
                return [model]
            elif args.net == 'tiramisu':
                from densenets import DenseNetFCN
                model = DenseNetFCN(input_shape)
                return [model]
            elif args.net == 'segcapsr1':
                from capsnet import CapsNetR1
                model_list = CapsNetR1(input_shape)
                return model_list
            elif args.net == 'segcapsr3':
                from capsnet import CapsNetR3
                model_list = CapsNetR3(input_shape, num_classes)
                return model_list
            elif args.net == 'capsbasic':
                from capsnet import CapsNetBasic
                model_list = CapsNetBasic(input_shape)
                return model_list
            elif args.net == 'isensee':
                from isensee import ResidualUnet2D
                model = ResidualUnet2D(input_shape, args.out_classes)
                return [model]
            else:
                raise Exception('Unknown network type specified: {}'.format(args.net))