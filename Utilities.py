"""
Functions for ease of use.

"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import skimage
import matplotlib.pyplot as plt
import keras as ks

def dcounter(defects):
    """
    Function for counting up all defects in a list, returned as a string (text).
    Used for titles and similar
    """
    u,c = np.unique(defects,return_counts=True)         # unique defects and counts
    txt = ', '.join([f'{k}: {v}' for k,v in zip(u,c)])  # summarize into a string
    return txt


def load_PVEL_images(path1 ='Data/PVEL-AD/trainval/JPEGImages',path2= 'Data/PVEL-AD/solar_cell_EL_image/PVELAD/EL2021/othertypes/good',SIZE=256):
    """
    A function that loads and resizes the images from the PVEL-dataset.
    """
    pvel = ks.utils.image_dataset_from_directory(path1,labels=None,batch_size=1,image_size=(SIZE,SIZE),color_mode='grayscale',shuffle=False)
    pvel = np.array(list(pvel.as_numpy_iterator())).reshape(4500,SIZE,SIZE,1)

    num = 11353 # (full)
    good_images = ks.utils.image_dataset_from_directory(path2,labels=None,batch_size=1,image_size=(SIZE,SIZE),color_mode='grayscale',shuffle=False)
    good_images = np.array(list(good_images.as_numpy_iterator())[:num]).reshape(num,SIZE,SIZE,1)

    pvel = np.concatenate([pvel,good_images]) #Transform the list to an array for ease of use, and puts them togheter
    return pvel


def selectionPVEL(df, classes=None, short=False):
    """
    
    short: int between 4500-14500
    """
    if classes is None:
        classes = ['black_core','crack','dislocation','finger','thick_line','anomaly-free','short_circuit']

    if short:
        df = df.iloc[:short]

    df.Label = df.Label.apply(lambda x: combine_defects(x))
    
    sdf = df[df.Label.map(lambda x:len(np.unique(x)) == 1)]
    sdf = sdf[sdf.Label.map(lambda x:x[0] in classes)]
    sdf.loc[:,'Label'] = sdf.Label.map(lambda x:x[0])

    return sdf


def combine_defects(row):
    row = ['crack' if x == 'star_crack' else x for x in row]
    row = ['dislocation' if x == 'horizontal_dislocation' else x for x in row]
    row = ['dislocation' if x == 'vertical_dislocation' else x for x in row]

    return row


def selection(lab, mx = 900):
    """Takes pd.Series object"""
    lab = lab.iloc[np.random.permutation(len(lab))]
    unq, c = np.unique(lab, return_counts=True)
    exc = pd.Series(dtype=float)
    sel = pd.Series(dtype=float)
    for i, u in enumerate(unq):
        if c[i] > mx:
            exc = pd.concat([exc, lab[lab==u][mx:]])
            sel = pd.concat([sel, lab[lab==u][:mx]])
        else:
            sel = pd.concat([sel, lab[lab==u]])
    return sel, exc


def selectionT(lab, mx = 900):
    """Takes pd.Series object"""
    lab = lab.iloc[np.random.permutation(len(lab.Label))]
    unq, c = np.unique(lab.Label, return_counts=True)
    exc = pd.DataFrame()
    sel = pd.DataFrame()
    for i, u in enumerate(unq):
        if c[i] > mx:
            exc = pd.concat([exc, lab[lab.Label==u][mx:]])
            sel = pd.concat([sel, lab[lab.Label==u][:mx]])
        else:
            sel = pd.concat([sel, lab[lab.Label==u]])
    return sel, exc


def small_cores(DF,mean_mult = 1/4):
    indices = DF.index
    DF = DF.reset_index()
    cores = DF[DF.Label.apply(lambda x:'black_core' in x)]
    #cores = cores.reset_index()
    Areas = pd.DataFrame(columns = [['Label','Area','OrigIndx']])
    for row_ind in cores.index:
        row = cores.loc[row_ind]
        size = row.Resolution
        width = 0
        height = 0
        for col_ind,defect in enumerate(row.Label):
            if defect == 'black_core':
                width += float(int(row.right[col_ind])/size) - float(int(row.left[col_ind])/size)
                height += float(int(row.bottom[col_ind])/size) - float(int(row.top[col_ind])/size)
        Areas.loc[row_ind] = ['black_core',width*height,indices[row_ind]]
    
    return Areas[(Areas.Area < Areas.Area.mean()*mean_mult).values]



def no_ticks(ax):
    """
    Shortcut for removing ticks from a matplotlib axis.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def rescale(x):
    """
    Function for normalizing the image to the range 0 to 1.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def addlabels(x,y, f=''):
    """
    Function for adding numbers to the bars in a barplot.
    """
    for i,x_ in enumerate(x):
        plt.text(x_, 0.70*y[i]/2,f'{y[i]:{f}}', ha = 'center', bbox = dict(facecolor = 'white', alpha = 0.5))


def permutation_shuffle(X, ys):
    """
    Function for shuffling multiple arrays in the same random order.
    """
    inds = np.random.permutation(len(X))
    for i, y in enumerate(ys):
        ys[i] = y[inds]
    return X[inds], ys, inds