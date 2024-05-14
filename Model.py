# This file will contain the architecture used in the Paper
import keras as ks 
from keras.models import Model
# from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout, Input, UpSampling2D, concatenate
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow_addons.image import gaussian_filter2d
from keras import backend as K
from sklearn.utils import class_weight

import numpy as np
import pandas as pd



def AugmentationBlock(inputs, operations, training=False, make_model=False):
    """
    Function for adding layers of preprocessing to the models.

    If make_model=True, inputs should be input_shape, otherwise it should be previous layer.

    """

    if make_model:
        inputs = Input(inputs)

    aug = inputs

    for layer in operations:
        aug = layer(aug, training=training)
    
    if make_model:
        model = Model(inputs, aug)
        return model

    return aug


# taken from: https://github.com/beresandras/image-augmentation-layers-keras/blob/master/augmentations.py
class RandomGaussianNoise(ks.layers.Layer):
    def __init__(self, stddev=0.02, p = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.proba = p

    def call(self, images, training=True):
        
        if training:
            if np.random.random() < self.proba:
                images = tf.clip_by_value(
                    images + tf.random.normal(tf.shape(images), stddev=self.stddev), 0, 1
                )
        return images
    


class RandomRot180(ks.layers.Layer):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.proba = p

    def call(self, images, training=True):
        
        if training:
            if np.random.random() < self.proba:
                images = tf.image.rot90(images,k=2)
        return images
    

class RandomSharpen(ks.layers.Layer):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.proba = p

    def call(self, images, training=True):
        if training:
            if np.random.random() < self.proba:
                images = sharpen(images)
        return images
    
def sharpen(image,seed=None):
    """
    Augmentation function that applies a sharpening/blurring filter to an image.
    """
    rn = 2
    rnin = np.random.randint(3,13)
    amount = 2
    return tf.clip_by_value(image + amount*(image-gaussian_filter2d(image,rnin,rn)), 0, 1)





def DefectRating(input_shape, num=None):
    """
    This is the fully-connected layers of the Defect-rating (0,0.33,0.67,1) model used in regression tasks
    It takes the output of a Convolutional layer as the Input, and outputs a value from 0 to 1
    """
    inputs = Input(shape=input_shape)
    mod = ks.layers.Flatten()(inputs)

    mod = Dropout(.6)(mod)
    mod = BatchNormalization()(mod)
    mod = Dense(32, activation='relu')(mod)
    mod = Dropout(.2)(mod)
    mod = BatchNormalization()(mod)
    output = Dense(1, activation='sigmoid')(mod)
    model = Model(inputs, output)
    return model

def DefectRatingT(input_shape, num=None):
    """
    This is the fully-connected layers of the Defect-rating (0,0.33,0.67,1) model used in regression tasks
    It takes the output of a Convolutional layer as the Input, and outputs a value from 0 to 1
    num unused, but 
    """
    input1 = Input(shape=input_shape)
    input2 = Input(shape=(1,))
    mod = GlobalAveragePooling2D()(input1)

    mod = Dropout(.6)(mod)
    mod = BatchNormalization()(mod)
    mod = Dense(32, activation='relu')(mod)
    mod = Concatenate(axis=1)([mod, input2])
    mod = Dropout(.2)(mod)
    mod = BatchNormalization()(mod)
    output = Dense(1, activation='sigmoid')(mod)
    model = Model([input1,input2], output)
    return model


def DefectRatingCls(input_shape, num_classes=4):
    """
    This is the fully-connected layers for the Defect-rating (0,0.33,0.67,1) model used in classification tasks.
    It takes the output of a Convolutional layer and the number of classes as the Input, and outputs a softmax of the classes
    """
    inputs = Input(shape=input_shape)

    mod = GlobalAveragePooling2D()(inputs)

    mod = BatchNormalization()(mod)
    output = Dense(num_classes, activation='softmax')(mod)
    model = Model(inputs, output)
    return model



def DefectRatingClsT(input_shape, num_classes=4):
    """
    This is the fully-connected layers for the Defect-rating (0,0.33,0.67,1) model used in classification tasks.
    It takes the output of a Convolutional layer and the number of classes as the Input, and outputs a softmax of the classes
    """
    input1 = Input(shape=input_shape)
    input2 = Input(shape=(1,))

    mod = GlobalAveragePooling2D()(input1)

    mod = BatchNormalization()(mod)
    mod = Dense(16, activation='relu')(mod)
    mod = Concatenate(axis=1)([mod, input2])
    mod = Dropout(.2)(mod)
    mod = BatchNormalization()(mod)
    output = Dense(num_classes, activation='softmax')(mod)
    model = Model([input1, input2], output)
    return model


def DefectType(input_shape, num_classes):
    """
    This is the fully-connected layers for the Defect-type (anomaly) model used in classification tasks.
    It takes the output of a Convolutional layer and the number of classes as the Input, and returns the softmax of the classes
    """
    inputs = Input(shape=input_shape)

    mod = GlobalAveragePooling2D()(inputs)

    output = Dense(num_classes, activation='softmax')(mod)
    model = Model(inputs, output)
    return model



def DefectTypeT(input_shape, num_classes):
    """
    This is the fully-connected layers for the Defect-type (anomaly) model used in classification tasks.
    It takes the output of a Convolutional layer and the number of classes as the Input, and returns the softmax of the classes
    """
    input1 = Input(shape=input_shape)
    input2 = Input(shape=(1,))

    mod = GlobalAveragePooling2D()(input1)

    mod = Dropout(.6)(mod)
    mod = BatchNormalization()(mod)
    mod = Dense(32, activation='relu')(mod)
    mod = Concatenate(axis=1)([mod, input2])
    mod = Dropout(.2)(mod)
    mod = BatchNormalization()(mod)
    output = Dense(num_classes, activation='softmax')(mod)
    model = Model([input1, input2], output)
    return model



def transfer_net(input_shape, LS=1, wght='imagenet'):
    """
    A feature extractor using a pre-built model through transfer learning.
    The function takes the expected shape of the image as input and outputs a model which will give the features of the image.
    """
    
    base_model = ks.applications.VGG16(
        weights=wght,
        input_shape=input_shape,
        include_top=False)
    
    n = int(len(base_model.layers)*(1-LS))
    for layer in base_model.layers[:n]:
        layer.trainable = False
    
    return base_model



def transfer_net__(input_shape, LS=1, wght='imagenet'):
    """
    A feature extractor using a pre-built model through transfer learning.
    The function takes the expected shape of the image as input and outputs a model which will give the features of the image.
    """
    inputs = Input(shape=input_shape)
    tfnets = [ks.applications.EfficientNetV2S, ks.applications.VGG16] #, ks.applications.RegNetX004, ks.applications.RegNetY004]
    outs = []
    for tfnet in tfnets:
        base_model = tfnet(
            weights=wght,
            input_shape=input_shape,
            include_top=False)
        
        n = int(len(base_model.layers)*(1-LS))
        for layer in base_model.layers[:n]:
            layer.trainable = False
        
        outs.append(Dense(1024, activation='relu')(GlobalAveragePooling2D()(base_model(inputs))))

    outputs = concatenate(outs, axis=1)
    model = Model(inputs, outputs)
    return model


def get_weights_():
    """
    4 classes
    A function that loads the weights used in CWCCLoss, where the different weights should correspond to how damaging that error should be.
    """
    weights =   [[  1,   53,   146,  238],
                 [  9,   1,   1.5,   13],
                 [  13,   1.5,   1,   9],
                 [ 323,   231,   138,   1]]
    weights=np.array([np.array(i) for i in weights])
    return weights


def get_weights():
    """
    3 classes
    A function that loads the weights used in CWCCLoss, where the different weights should correspond to how damaging that error should be.
    """
    weights =   [[    1, 21.86, 49.74],
                 [ 3.01,     1,  3.01],
                 [12.98,  9.98,     1]]
    
    weights=np.array([np.array(i) for i in weights])
    return weights


def CWCCLoss(y_true, y_pred):
    """
    This is a weighted Categorical_crossentropy loss using a weight based on the prediction and true class label.
    """
    weights = tf.constant(get_weights(),dtype='float16')
    ind1 = K.argmax(y_true,axis=1)
    ind2 = K.argmax(y_pred,axis=1)
    ind = tf.transpose([ind2,ind1])
    w = tf.gather_nd(weights,ind)

    return K.sum(w * ks.losses.CategoricalCrossentropy(reduction='none')(y_true,y_pred)/tf.cast(tf.size(w),dtype='float16'))


def CostMetric(y_true, y_pred):
    weights = tf.constant(get_weights(),dtype='float16')
    ind1 = K.argmax(y_true,axis=1)
    ind2 = K.argmax(y_pred,axis=1)
    ind = tf.transpose([ind2,ind1])
    test = tf.gather_nd(weights,ind)
    return K.sum(test) / tf.cast(tf.shape(y_true)[0], dtype='float16')


def modeling_unit(X, y, conv_layers, FC, loss, metrics, BATCH_SIZE=32, EPOCHS=5, CB=None, save_history=None, name=None, LR=0.001, clswght=None, val_dat=None):
    """
    This function simplifies the training of a model used in run, and trains a model consisting of a convolutional part and a fully-connected part.
    """


    y_integers = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight = clswght, classes = np.unique(y_integers), y = y_integers)
    d_class_weights = dict(enumerate(class_weights))

    model = Model(conv_layers.input, FC(conv_layers.output))
    model.compile(optimizer=Adam(learning_rate=LR), loss=loss, metrics=metrics)
    hist = model.fit(X, y, batch_size=BATCH_SIZE, validation_split=0.2, validation_data=val_dat, 
                     epochs=EPOCHS, callbacks=CB, class_weight=d_class_weights
                     )

    if save_history is not None:
        h = pd.DataFrame(hist.history)
        if not name:
            name = f'Model{1 + int(len(save_history.columns)/len(h.columns))}'
        save_history[[name+x for x in h.columns]] = h

    return model


def modeling_unitT(X, y, conv_layers, FC, loss, metrics, BATCH_SIZE=32, EPOCHS=5, MCC=None, save_history=None, name=None, LR=0.001, clswght=None, val_dat=None):
    """
    This function simplifies the training of a model used in run, and trains a model consisting of a convolutional part and a fully-connected part. 
    """
    if MCC:
        MCC = [MCC]

    inds = np.random.permutation(len(y))
    X[0] = X[0][inds]
    X[1] = X[1].iloc[inds]
    y = y.iloc[inds]

    y_integers = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight = clswght, classes = np.unique(y_integers), y = y_integers)
    d_class_weights = dict(enumerate(class_weights))

    input1 = Input(shape=conv_layers.input.shape[1:])
    input2 = Input(shape=(1,))
    mod = conv_layers(input1)
    out = FC([mod, input2])

    model = Model([input1,input2], out)
    model.compile(optimizer=LR, loss=loss, metrics=metrics)
    hist = model.fit(X, y, batch_size=BATCH_SIZE, validation_split=0.2, validation_data=val_dat, 
                     epochs=EPOCHS, callbacks=MCC, class_weight=d_class_weights
                     )

    if save_history is not None:
        h = pd.DataFrame(hist.history)
        if not name:
            name = f'Model{1 + int(len(save_history.columns)/len(h.columns))}'
        save_history[[name+x for x in h.columns]] = h
        
    return model




if __name__ == '__main__':
    input_shape = (100,100,3)

    master = transfer_net(input_shape)
    dtype =  DefectType(master.output.shape[1:],7)

    drating = DefectRatingCls(master.output.shape[1:],4)
    model = Model(master.input, drating(master.output))

    print(model.summary())
