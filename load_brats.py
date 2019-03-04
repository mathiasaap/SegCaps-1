import numpy as np
from os.path import join, basename
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def convert_brats_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False, num_classes=4):
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

    #mean = np.array([18.426106306720985, 24.430354760142666, 24.29803657467962, 19.420110564555472])
    #std = np.array([104.02684046042094, 136.06477850668273, 137.4833895418739, 109.29833288911334])

    brats_min = np.array([0.0, 0.0, 0.0, 0.0])
    brats_max = np.array([6476.0, 9751.0, 11737.0, 5337.0])

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

        for i in range(4):
            img[img[:,:,:,i] > brats_max[i]] = brats_max[i]
            img[img[:,:,:,i] < brats_min[i]] = brats_min[i]

        img += -brats_min
        img /= (brats_max + -brats_min)


        if not no_masks:
            itk_mask = sitk.ReadImage(join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)

            label = mask.astype(np.int64)
            if num_classes == 1:
                mask_onehot = label.reshape(240,240,-1,1)
            else:
                mask_onehot = np.eye(num_classes)[label]


        try:
            show_modal = 0
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3, show_modal], cmap='gray')
            if not no_masks:
                ax[0].imshow(mask[:, :, img.shape[2] // 3], alpha=0.7, cmap='Reds')
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2, show_modal], cmap='gray')
            if not no_masks:
                ax[1].imshow(mask[:, :, img.shape[2] // 2], alpha=0.7, cmap='Reds')
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4, show_modal], cmap='gray')
            if not no_masks:
                ax[2].imshow(mask[:, :, img.shape[2] // 2 + img.shape[2] // 4], alpha=0.7, cmap='Reds')
            ax[2].set_title('Slice {}/{}'.format(img.shape[2] // 2 + img.shape[2] // 4, img.shape[2]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle(fname)
            plt.savefig(join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print('\n'+'-'*100)
            print('Error creating qualitative figure for {}'.format(fname))
            print(e)
            print('-'*100+'\n')

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask_onehot)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask_onehot
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)
