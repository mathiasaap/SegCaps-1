import numpy as np
from os.path import join, basename
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def convert_heart_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False):
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

    heart_min = 0
    heart_max = 2196.0
    mean = 0 #np.array([170.25972919418757])
    std = 1 #np.array([257.885508476468])

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
        img = np.rollaxis(img, 0, 3) 
        
        #img -= mean
        #img /= std
        
        img[img > heart_max] = heart_max
        img[img < heart_min] = heart_min
        img += -heart_min
        img /= (heart_max + -heart_min)
        
        #img = img[:, :, :, 3] # Select only t1w during initial testing
        #img = (img-img.mean())/img.std()
        
        
        if not no_masks:
            itk_mask = sitk.ReadImage(join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)
            #mask[mask < 0.5] = 0 # Background
            #mask[mask > 0.5] = 1 # Edema, Enhancing and Non enhancing tumor
            
            label = mask.astype(np.int64)
            #print(label[150])
            masks = np.eye(2)[label] #label.reshape(320,320,-1,1)
            print("Created mask shape: {}".format(masks.shape))
            #mask = masks.astype(np.float32)
            
            #masks[masks>0.5] = one_hot_max
            #masks[masks<0.5] = 1-one_hot_max
            mask = masks
            #mask = masks.astype(np.uint8)
            

        try:
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3], cmap='gray')
            if not no_masks:
                ax[0].imshow(mask[:, :, img.shape[2] // 3, 0], alpha=0.40, cmap='Reds')
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
            if not no_masks:
                ax[1].imshow(mask[:, :, img.shape[2] // 2, 0], alpha=0.40, cmap='Reds')
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4], cmap='gray')
            if not no_masks:
                ax[2].imshow(mask[:, :, img.shape[2] // 2 + img.shape[2] // 4, 0], alpha=0.40, cmap='Reds')
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