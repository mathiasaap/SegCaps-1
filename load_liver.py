import numpy as np
from os.path import join, basename
import SimpleITK as sitk
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def create_color_map(dataset):
    green = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    
    red = {'red':   ((0.0, 0.0, 0.0),
               (1.0, 1.0, 1.0)),

     'green': ((0.0, 0.0, 0.0),
               (1.0, 0.0, 0.0)),

     'blue':  ((0.0, 0.0, 0.0),
               (1.0, 0.0, 0.0))}
        
    blue = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0))}
    
    colors = ["Greys"]
    if dataset == "brats":
        colors.append(LinearSegmentedColormap('Blue', blue))
        colors.append(LinearSegmentedColormap('Red', red))
        colors.append(LinearSegmentedColormap('Green', green))
        
    else:
        colors.append(LinearSegmentedColormap('Green', green))
        colors.append(LinearSegmentedColormap('Red', red))
        colors.append(LinearSegmentedColormap('Blue', blue))
    return colors

def convert_liver_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False, num_classes=3):
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

    #Min: [-2048.0]
    #Max: [4009.0]

    ct_min = -65.0
    ct_max = 185.0
    #Spleen Min: [-1024.0]
    #Spleen Max: [3072.0]
    #mean = np.array([-541.1801174550513])
    #std = np.array([492.2428379436813])

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

        img[img > ct_max] = ct_max
        img[img < ct_min] = ct_min
        img += -ct_min
        img /= (ct_max + -ct_min)

        #img = img[:, :, :, 3] # Select only t1w during initial testing
        #img = (img-img.mean())/img.std()


        if not no_masks:
            itk_mask = sitk.ReadImage(join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)
            #mask[mask < 0.5] = 0 # Background
            #mask[mask > 0.5] = 1 # Edema, Enhancing and Non enhancing tumor

            label = mask.astype(np.int64)
            if num_classes == 1:
                masks = label.reshape(512,512,-1,1)
            else:
                masks = np.eye(num_classes)[label]
            print("Created mask shape: {}".format(masks.shape))
            #mask = masks.astype(np.float32)

            #masks[masks>0.5] = one_hot_max
            #masks[masks<0.5] = 1-one_hot_max
            mask = masks
            #mask = masks.astype(np.uint8)


        try:
            colors = create_color_map("liver")
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3], cmap='gray')
            if not no_masks:
                masking1 = mask[:, :, img.shape[2] // 3, 1]
                masking1 = np.ma.masked_where(masking1 == 0, masking1)
                ax[0].imshow(masking1, alpha=0.4, cmap=colors[1],vmin=0, vmax=1)
                
                masking2 = mask[:, :, img.shape[2] // 3, 2]
                masking2 = np.ma.masked_where(masking2 == 0, masking2)
                ax[0].imshow(masking2, alpha=0.4, cmap=colors[2],vmin=0, vmax=1)
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
            if not no_masks:
                masking1 = mask[:, :, img.shape[2] // 2, 1]
                masking1 = np.ma.masked_where(masking1 == 0, masking1)
                ax[1].imshow(masking1, alpha=0.4, cmap=colors[1],vmin=0, vmax=1)
                
                masking2 = mask[:, :, img.shape[2] // 2, 2]
                masking2 = np.ma.masked_where(masking2 == 0, masking2)
                ax[1].imshow(masking2, alpha=0.4, cmap=colors[2],vmin=0, vmax=1)
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4], cmap='gray')
            if not no_masks:
                masking1 = mask[:, :, img.shape[2] // 2 + img.shape[2] // 4, 1]
                masking1 = np.ma.masked_where(masking1 == 0, masking1)
                ax[2].imshow(masking1, alpha=0.4, cmap=colors[1],vmin=0, vmax=1)
                
                masking2 = mask[:, :, img.shape[2] // 2 + img.shape[2] // 4, 2]
                masking2 = np.ma.masked_where(masking2 == 0, masking2)
                ax[2].imshow(masking2, alpha=0.4, cmap=colors[2],vmin=0, vmax=1)
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
