from skimage.util import random_noise
from skimage.filters import unsharp_mask
import random as rd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from Utilities import *
from Model import AugmentationBlock

def targeted_auggen(images, labels, augs, mn=10**5):
    if not augs:
        return images, labels
    
    datagen = AugmentationBlock(images[0].shape, augs, make_model=True)

    co = np.unique(labels, return_counts=True)[1]
    num = min(co.max(), mn)
    unique_labs = np.unique(labels)
    labels = labels.reset_index().iloc[:,-1]
    #origIndx = [i for i in range(len(images))]
    for i, c in enumerate(co):
        if c >= num:
            continue
        inds = labels[labels==unique_labs[i]].index
        tmp_inds = np.random.choice(inds, size=num-len(inds), replace=True)
        tmp_labels = labels.iloc[tmp_inds]
        tmp_imgs = datagen(images[tmp_inds])

        images = np.concatenate([images,tmp_imgs])
        labels = pd.concat([labels, tmp_labels]).reset_index().iloc[:,-1]
        #origIndx += tmp_inds.tolist()
    inds = np.random.permutation(len(images))
    return images[inds], labels.iloc[inds].reset_index(drop=True) #, np.array(origIndx)[inds]


def random_auggen(images, labels, augs, num=1000):
    if not augs:
        return images, labels
    
    datagen = AugmentationBlock(images[0].shape, augs, make_model=True)
    labels = labels.reset_index(drop=True)

    samples = np.random.choice(labels.index, num)
    labels = pd.concat([labels, labels.iloc[samples]]).reset_index(drop=True)
    images = np.concatenate([images, datagen(images[samples])])

    inds = np.random.permutation(len(images))
    return images[inds], labels.iloc[inds].reset_index(drop=True) #, samples


# Percentage chance of augmentation
def random_augmentation(image,operation,percentage):
    """
    Module for augmenting images with a set chance.
    It will take an image, the augmentation technique and the percentage chance of using the augmentation.   
    """
    if rd.randint(0,100) < percentage:
        return operation(image), True
    else:
        return image, False


def random_mix_augmentation(image, images, operation, percentage):
    """
    Module for augmenting images with a set chance.
    It will take an image, the augmentation technique and the percentage chance of using the augmentation.   
    """
    if rd.randint(0,100) < percentage:
        return operation(image, images), True
    else:
        return image, False


# Geometric augmentation
def rot180(image, seed=None):
    """
    Augmentation function that applies 180 degree rotations to an image
    """
    return tf.image.rot90(image,k=2).numpy()


def flip_left_right(image, seed=None):
    """
    Augmentation function that applies flipping around the vertical axis on an image.
    """
    return tf.image.flip_left_right(image).numpy()


def flip_up_down(image, seed=None):
    """
    Augmentation function that applies flipping around the horizontal axis on an image.
    """
    return tf.image.flip_up_down(image).numpy()


def transpose(image, seed=None):
    """
    Augmentaion function that applies transposing to an image.
    """
    return tf.image.transpose(image).numpy()


# Pixel augmentation
def brightness(image,max_delta=0.15):
    """
    Augmentation function that adjusts the brightness of an image from -max_delta to max_delta.
    """
    return tf.image.random_brightness(image,max_delta).numpy()


def contrast(image, low=1.3, up=1.4):
    """
    Augmentation function that increases the ontrast of an image from low to high.
    """
    return tf.image.random_contrast(image, low, up).numpy()


def shrp_blur(image,seed=None):
    """
    Augmentation function that applies a sharpening/blurring filter to an image.
    """
    rn = np.random.random()*1 - 0.5
    rnin = np.random.randint(1,10)
    return unsharp_mask(image,radius=rnin,amount=rn)


def add_noise(image, mu=0,var=0.15):
    """
    Augmentation function that adds noise to an image from a normal distribution with mean equal to mu and standard-variation equal to var.
    """
    x,y,z = image.shape
    noise = np.random.normal(mu,var,x*y*z)
    image = image.flatten() + noise
    image = image.reshape(x,y,z)
    return image


def noise(image, mean=0, var=0.005):
    """
    Augmentation function using the skimage.util.random_noise function to add noise to an image from 
    a normal distribution with mean equal to mean and standard-deviation equal to var.
    """
    seed = np.random.randint(10000)
    return random_noise(image,'gaussian',seed,mean=mean,var=var)


# Mixing and erasing
def mixup(image,images):
    shape = image.shape[:2]
    first_image = image.copy()
    ind = np.random.randint(0,len(images))
    second_image = images[ind]    
    new_image = np.add(first_image,second_image)/2.0
    return new_image

def box(image, boxsize=50):
    nimg = image.copy()
    shape = nimg.shape[:2]
    x = np.random.randint(int(boxsize/2), shape[0]-int(boxsize/2))
    y = np.random.randint(int(boxsize/2), shape[1]-int(boxsize/2))
    nimg[y-int(boxsize/2):y+int(boxsize/2), x-int(boxsize/2):x+int(boxsize/2)] = rescale(np.random.normal(size=(boxsize,boxsize)).reshape(boxsize,boxsize,1))

    return nimg
    
def rectangle_box(image, xsize=None, ysize=None):
    new_image = image.copy()
    shape = image.shape[:2]

    if xsize is None:
        xsize = np.random.randint(int(shape[1]*0.1),int(shape[1]*0.4))

    if ysize is None:
        ysize = np.random.randint(int(shape[0]*0.1),int(shape[0]*0.4))

    x_input = int(xsize/2)
    y_input = int(ysize/2)

    y = np.random.randint(int(ysize/2),shape[0]-int(ysize/2))
    x = np.random.randint(int(xsize/2),shape[1]-int(xsize/2))
    new_image[y-y_input:y+y_input,x-x_input:x+x_input] = rescale(np.random.normal(size=(2*y_input,2*x_input)).reshape(2*y_input,2*x_input,1))
    return new_image

def mix_rectangle_box(image, images, xsize=None, ysize=None):
    new_image = image.copy()
    shape = image.shape[:2]

    if xsize is None:
        xsize = np.random.randint(int(shape[1]*0.1),int(shape[1]*0.4))

    if ysize is None:
        ysize = np.random.randint(int(shape[0]*0.1),int(shape[0]*0.4))

    y = np.random.randint(int(ysize/2),shape[0]-int(ysize/2))
    x = np.random.randint(int(xsize/2),shape[1]-int(xsize/2))

    ind = np.random.randint(len(images))
    new_image[y-int(ysize/2):y+int(ysize/2),x-int(xsize/2):x+int(xsize/2)]+=images[ind][y-int(ysize/2):y+int(ysize/2),x-int(xsize/2):x+int(xsize/2)]
    new_image[y-int(ysize/2):y+int(ysize/2),x-int(xsize/2):x+int(xsize/2)]/=2
    return new_image


def adjust_pixels(image):
    """
    Augmentation function that adjusts the pixels of an image to stay within the range og 0 to 1.
    """
    image = np.where(image>1,1,image)
    image = np.where(image<0,0,image)
    return image


def make_augimgs(images,labels_df,operations,num = 1000,percentage = 50,shape = (256,256)):
    """
    Function for creating multiple augmented images of a given list of images.
    It will select (num) amount of images randomly and apply augmentations given by operations.
    It will return a list of new augmented images together with their labels and an index of original images.
    """
    train_images = []
    train_labels = []
    indexes = np.random.choice(labels_df.index, num)
    for ind,image in enumerate(images[indexes]):
        new_image = image.copy()
        augs = 0
        for operation in operations:
            if 'mix' in operation.__name__:
                new_image, aug = random_mix_augmentation(new_image.reshape(shape[0],shape[1],1), images, operation, percentage)
            else:
                new_image, aug = random_augmentation(new_image.reshape(shape[0],shape[1],1), operation, percentage)
            
            new_image = adjust_pixels(new_image)
            augs += aug
        
        if augs == 0:
            operation = np.random.choice(operations)
            if 'mix' in operation.__name__:
                new_image = adjust_pixels(random_mix_augmentation(new_image.reshape(shape[0],shape[1],1), images, operation, 100)[0])
            else:
                new_image = adjust_pixels(random_augmentation(new_image.reshape(shape[0],shape[1],1), operation, 100)[0])

        
        train_images.append(rescale(new_image))
        train_labels.append(labels_df.loc[indexes[ind]])

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    DFaug = pd.DataFrame()
    DFaug['Label'] = train_labels
    DFaug['OrigIndx'] = indexes
    return train_images,DFaug


def targeted_aug(images, labels, operations, mn=10**5, size=256):
    if not operations:
        return images, labels
    
    co = np.unique(labels,return_counts=True)[1]
    num = min(co.max(), mn)
    unique_probas = np.unique(labels)
    labels = labels.reset_index().iloc[:,-1]
    for i, c in enumerate(co):
        if c >= num:
            continue
        inds = labels[labels==unique_probas[i]].index
        tmp_imgs, tmp_augs = make_augimgs(images[inds], labels.loc[inds].reset_index().iloc[:,-1], operations,
                                        num=num-len(inds), shape=(size,size))
        tmp_augs = tmp_augs.Label
        images = np.concatenate([images,tmp_imgs])
        labels = pd.concat([labels,tmp_augs]).reset_index().iloc[:,-1]
    return images, labels


def targeted_augT(images, labels, operations, mn=10**5, size=256):
    if not operations:
        return images, labels
    
    co = np.unique(labels.Label,return_counts=True)[1]
    num = min(co.max(), mn)
    unique_probas = np.unique(labels.Label)
    labels = labels.reset_index().iloc[:,-2:]
    for i, c in enumerate(co):
        if c >= num:
            continue
        inds = labels[labels.Label==unique_probas[i]].index
        tmp_imgs, tmp_augs = make_augimgs(images[inds], labels.loc[inds].reset_index().Label, operations,
                                        num=num-len(inds), shape=(size,size,1))
        tmp_augs['Type'] = labels.loc[inds].Type.to_numpy()[tmp_augs.OrigIndx]
        tmp_augs = tmp_augs[['Label', 'Type']]
        images = np.concatenate([images,tmp_imgs])
        labels = pd.concat([labels,tmp_augs]).reset_index()[['Label','Type']]
    return images, labels


def aug_dataset(images, labels, operations, num=1000, percentage=50, SIZE=256):
    """
    A function that makes the augmentations on the dataset, and concatenates them togheter (stack/add).
    It returns the non-augmented images togheter with the augmentd images, and their corresponding labels.
    """
    if not operations:
        return images, labels
    
    train_images, DFaug = make_augimgs(images, labels, operations, num=num, percentage=percentage,shape=(SIZE,SIZE))
    imgs_withaug = np.concatenate([images,train_images])
    labels_withaug = pd.concat([labels,DFaug.Label])
    return imgs_withaug, labels_withaug


def aug_datasetT(images, labels, operations, num=1000, percentage=50, SIZE=256):
    """
    A function that makes the augmentations on the dataset, and concatenates them togheter (stack/add).
    It returns the non-augmented images togheter with the augmentd images, and their corresponding labels.
    """
    if not operations:
        return images, labels
    
    train_images, DFaug = make_augimgs(images, labels.Label, operations, num=num, percentage=percentage,shape=(SIZE,SIZE))
    DFaug['Type'] = labels.Type.to_numpy()[DFaug.OrigIndx]
    DFaug = DFaug[['Label', 'Type']]
    imgs_withaug = np.concatenate([images, train_images])
    labels_withaug = pd.concat([labels,DFaug]).reset_index()[['Label', 'Type']]
    return imgs_withaug, labels_withaug


def augment_small_cores(images,DF,operations,mean_mult = 1/4,num = 1000,percentage = 50,shape = (256,256)):
    if not operations:
        return np.empty(shape=(1, shape[0], shape[0], 1)), pd.DataFrame(columns=['Label'], dtype=str)
    
    small_BC = small_cores(DF,mean_mult=mean_mult)
    BC_images = images[small_BC.index]
    BC_df = small_BC.reset_index().Label.squeeze()
    img,labels = make_augimgs(BC_images,BC_df,operations,num=num, percentage=percentage,shape=shape)
    return img, labels


def visualize_augimgs(orig_images,augimgs,DFaug,name = 'Geometric'):
    """
    A function that plots different augmentations for visual inspection.
    """
    name = name + ' Augmentations'
    fig,ax = plt.subplots(2,5,figsize=(15,6))
    fig.suptitle(name)

    indexes = DFaug.OrigIndx
    inds = np.random.randint(0,len(indexes),5)

    ax[0][0].set_ylabel('Original Images')
    ax[1][0].set_ylabel('Augmented Images')
    for i in range(5):
        ax[0][i].imshow(orig_images[indexes[inds[i]]], vmin=0, vmax=1)
        no_ticks(ax[0][i])
        ax[0][i].set_title(DFaug.loc[inds[i]].Label)

    for i in range(5):
        ax[1][i].imshow(augimgs[inds[i]], vmin=0, vmax=1)
        no_ticks(ax[1][i])


    plt.show()