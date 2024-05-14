"""
Orion run-script.
"""

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import keras as ks
from keras.models import Model
# from keras.metrics import F1Score

from sklearn.metrics import f1_score, confusion_matrix
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
import time

from Utilities import *
from Augmentation import *
from Model import *

from Data.ELPV.elpv_reader import load_dataset # From ELPV GitHub

ks.mixed_precision.set_global_policy("mixed_float16")
tf.keras.utils.set_random_seed(66_6_22_88)

# ------------------- GLOBAL PARAMETERS ---------------------------
SIZE = 256
BATCH_SIZE = 64
EPOCHS = [60,60,60,60]
ELPV_TRAIN_TEST_SPLIT = 0.2
PVEL_TRAIN_TEST_SPLIT = 0.2
OP_NAMES = ['Base', 'Geometric','Pixel','All']
CLSWGHTS = 'balanced'

PATH_FC = 'tmp/elpv_w/checkpoint.h5'
PATH_CH1 = 'tmp/master1/checkpoint.h5'
PATH_CH2 = 'tmp/master2/checkpoint.h5'
PATH_CH3 = 'tmp/master3/checkpoint.h5'
PATH_CH4 = 'tmp/master4/checkpoint.h5'

LR = [0.000075, 0.0001, 0.000075, 0.000075]


reduce_lr = ks.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.9,
                            patience=3, min_lr=0.0000001, verbose=1)

pixel = [RandomSharpen(),RandomContrast((0,0.5)), RandomBrightness((-0.1,0.1), value_range=(0,1)), 
         RandomGaussianNoise(0.02)]
geometric = [RandomRot180(), RandomFlip("horizontal_and_vertical"), RandomRotation(0.01)]

augs = [None, geometric, pixel, geometric+pixel]

# ------------------------------------- ELPV ------------------------------------
elpv_imgs, elpv_prob, elpv_type = load_dataset('Data/ELPV/labels.csv')
elpv_imgs = elpv_imgs.reshape(len(elpv_imgs), 300, 300, 1)
elpv_imgs = tf.image.resize(elpv_imgs,(SIZE, SIZE)).numpy()
elpv_imgs = elpv_imgs/255

elpv_lab = np.unique(elpv_prob)

elpv_imgs, elpv_prob, inds = permutation_shuffle(elpv_imgs, [elpv_prob])
elpv_prob = elpv_prob[0]

elpv_prob = np.where(elpv_prob == elpv_lab[1], 0.5, elpv_prob)
elpv_prob = np.where(elpv_prob == elpv_lab[2], 0.5, elpv_prob)
ELPV_numclasses = len(np.unique(elpv_prob))

elpv_prob = pd.get_dummies(elpv_prob)
elpv_imgs = np.squeeze(gray2rgb(elpv_imgs))
test_split = int(elpv_imgs.shape[0]*0.2)
test_inds = inds[-test_split:]

elpv_test_prob = elpv_prob.iloc[-test_split:].reset_index(drop=True)
elpv_test_imgs = elpv_imgs[-test_split:]
elpv_prob = elpv_prob.iloc[:-test_split].reset_index(drop=True)
elpv_imgs = elpv_imgs[:-test_split]

val_split = int(elpv_imgs.shape[0]*0.2)
elpv_val_prob = elpv_prob.iloc[-val_split:].reset_index(drop=True)
elpv_val_imgs = elpv_imgs[-val_split:]
elpv_prob = elpv_prob.iloc[:-val_split].reset_index(drop=True)
elpv_imgs = elpv_imgs[:-val_split]


# ---------------------------- PVEL-AD -------------------------------------------


# Loading PVEL labels from pickle
DF = pd.read_pickle('Data/PVEL-AD/pandas_annotations') # Get the data form the images
class_names = np.loadtxt('Data/PVEL-AD/annotation_classes.txt',dtype=str) # Get the class names (defect names)

# Loading PVEL Images
pvel = load_PVEL_images(SIZE=SIZE)

pvel_labels = selectionPVEL(DF, short=6000).Label   # Gathers some classes into one and discards classes with too few samples 
                                        # as well as samples with more than one unique defect
pvel = pvel[pvel_labels.index] # Remove images with the same conditions as above
pvel = pvel/255

pvel = np.squeeze(gray2rgb(pvel))

# Train(0.8)/Test(0.2) split 
train_num = int(len(pvel_labels)*PVEL_TRAIN_TEST_SPLIT)

#pvelDF = pd.DataFrame()
#pvelDF['Label'] = pvel_labels
#pvelDF['Type'] = np.ones(len(pvel_labels))
#pvel_labels = pvelDF

pvel_inds = np.random.choice(pvel_labels.reset_index().index,train_num,replace=False)
pvel_test = pvel[pvel_inds]
pvel_train = np.delete(pvel,pvel_inds,axis=0)

pvel_test_labels = pvel_labels.iloc[pvel_inds]
pvel_test_labels = pvel_test_labels.reset_index().Label
pvel_train_labels = pvel_labels.drop(index = pvel_labels.index[pvel_inds])

pvel_train, pvel_train_labels, inds = permutation_shuffle(pvel_train, [pvel_train_labels.values])
pvel_train_labels = pvel_train_labels[0]

train_DF = DF.loc[pvel_labels.index].drop(index = pvel_labels.index[pvel_inds])
PVEL_numclasses = len(np.unique(pvel_train_labels))



val_split = int(pvel_train.shape[0]*0.2)
pvel_val_labels = pvel_train_labels[-val_split:]
pvel_val = pvel_train[-val_split:]
pvel_train_labels = pvel_train_labels[:-val_split]
pvel_train = pvel_train[:-val_split]

del pvel





# ----------------------------- Results Storage -------------------------------

# ELPV predictions
ELPVpred = pd.DataFrame()
ELPVpred['True'] = np.argmax(elpv_test_prob, axis=1)
ELPVpred['Inds'] = test_inds

# PVEL predictions
PVELpred = pd.DataFrame()
PVELpred['True'] = np.argmax(pd.get_dummies(pvel_test_labels), axis=1)
PVELpred['Inds'] = pvel_inds

# Cross-predictions
# Defect type predicted on ELPV data
ELPVXPred = pd.DataFrame()
ELPVXPred['True'] = np.argmax(elpv_test_prob, axis=1)
ELPVXPred['Inds'] = test_inds

# Defect rating predicted on PVEL data
PVELXPred = pd.DataFrame()
PVELXPred['True'] = np.argmax(pd.get_dummies(pvel_test_labels), axis=1)
PVELXPred['Inds'] = pvel_inds


Train_times = pd.DataFrame(index = ['elpv', 'pvel', 'elpv2','elpv_CWCC'],columns = OP_NAMES)
Train_times['EPOCHS'] = [EPOCHS for _ in Train_times.index]

Scores = pd.DataFrame(index = ['elpv', 'pvel', 'elpv2','elpv_CWCC'],columns = OP_NAMES)


ELPV_hist = pd.DataFrame()
PVEL_hist = pd.DataFrame()



# ------------------------ RUN ---------------------------



for i, augmentation in enumerate(augs):
# ------------------------------ Checkpoints -----------------------------------------
    mcc1 = ks.callbacks.ModelCheckpoint(
        filepath= PATH_CH1,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)
    mcc2 = ks.callbacks.ModelCheckpoint(
        filepath= PATH_CH2,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)
    mcc3 = ks.callbacks.ModelCheckpoint(
        filepath= PATH_CH3,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)
    mcc4 = ks.callbacks.ModelCheckpoint(
        filepath= PATH_CH4,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)

    
    conv = transfer_net((SIZE,SIZE,3))
    if augmentation:
        datagen = AugmentationBlock((SIZE,SIZE,3), augmentation, make_model=True)
        conv = Model(datagen.input,conv(datagen.output))

# ------------------------------ ELPV Crossentropy -----------------------------------

    # aug_imgs, aug_proba = targeted_auggen(elpv_imgs, pd.Series(np.argmax(elpv_prob, axis=1)), augmentation, mn=900)
    aug_imgs, aug_proba = random_auggen(elpv_imgs, pd.Series(np.argmax(elpv_prob, axis=1)), None, num=800)



    fc = DefectRatingCls(conv.output.shape[1:], ELPV_numclasses)

    start = time.time()
    model = modeling_unit(aug_imgs, pd.get_dummies(aug_proba), conv, fc, "categorical_crossentropy", 
                            ['accuracy', F1Score(num_classes = ELPV_numclasses,average='macro'), CostMetric],val_dat = (elpv_val_imgs,elpv_val_prob), 
                            CB=[mcc1,reduce_lr], save_history=ELPV_hist, name=f'{OP_NAMES[i]}_',
                            BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS[0], LR=LR[0], clswght=CLSWGHTS)
    Train_times.loc['elpv', OP_NAMES[i]] = (time.time() - start)/60
    

    print(model.optimizer.learning_rate)
    model.load_weights(PATH_CH1)

    score = F1Score(num_classes = ELPV_numclasses,average = 'macro')
    score.update_state(elpv_test_prob,model.predict(elpv_test_imgs))
    Scores.loc['elpv',OP_NAMES[i]]=score.result().numpy()

    ELPVpred[f'{OP_NAMES[i]}'] = list(model.predict(elpv_test_imgs))
    PVELXPred[f'{OP_NAMES[i]}'] = list(model.predict(pvel_test))

    fc.save_weights(PATH_FC)

# ------------------------------ PVEL -----------------------------------

    # aug_imgs, aug_proba = targeted_auggen(pvel_train, pd.Series(pvel_train_labels), augmentation, mn=800)
    aug_imgs, aug_proba = random_auggen(pvel_train, pd.Series(pvel_train_labels), None, num=1500)


    fc = DefectType(conv.output.shape[1:], PVEL_numclasses)

    start = time.time()
    model = modeling_unit(aug_imgs, pd.get_dummies(aug_proba), conv, fc, "categorical_crossentropy", 
                                ['accuracy',F1Score(num_classes = PVEL_numclasses,average='macro')], CB=[mcc2,reduce_lr], save_history=PVEL_hist, name= f'{OP_NAMES[i]}_',
                                 EPOCHS=EPOCHS[1], BATCH_SIZE=BATCH_SIZE, LR=LR[1],val_dat = (pvel_val,pd.get_dummies(pvel_val_labels))
                                 )
    Train_times.loc['pvel', OP_NAMES[i]] = (time.time() - start)/60
    model.load_weights(PATH_CH2)

    Scores.loc['pvel',OP_NAMES[i]]=f1_score(np.argmax(pd.get_dummies(pvel_test_labels).values*1,axis=1),
                                            np.argmax(model.predict(pvel_test),axis=1),average='macro')

    PVELpred[f'{OP_NAMES[i]}'] = list(model.predict(pvel_test))
    ELPVXPred[f'{OP_NAMES[i]}'] = list(model.predict(elpv_test_imgs))




# ------------------------------ ELPV Crossentropy 2 -----------------------------------

    # aug_imgs, aug_proba = targeted_auggen(elpv_imgs, pd.Series(np.argmax(elpv_prob, axis=1)), augmentation, mn=900)
    aug_imgs, aug_proba = random_auggen(elpv_imgs, pd.Series(np.argmax(elpv_prob, axis=1)), None, num=800)


    fc = DefectRatingCls(conv.output.shape[1:], ELPV_numclasses)

    fc.load_weights(PATH_FC)

    for layer in conv.layers:
        layer.trainable = False
        try:
            for la in layer.layers:
                la.trainable = False
        except:
            AttributeError

    start = time.time()
    model = modeling_unit(aug_imgs, pd.get_dummies(aug_proba), conv, fc, "categorical_crossentropy",
                            ['accuracy', F1Score(num_classes = ELPV_numclasses,average='macro'), CostMetric],val_dat = (elpv_val_imgs,elpv_val_prob), 
                            CB=[mcc3,reduce_lr], save_history=ELPV_hist, name=f'2{OP_NAMES[i]}_',
                            BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS[2], LR=LR[2], clswght=CLSWGHTS)

    Train_times.loc['elpv2', OP_NAMES[i]] = (time.time() - start)/60
    model.load_weights(PATH_CH3)
    
    score = F1Score(num_classes = ELPV_numclasses,average = 'macro')
    score.update_state(elpv_test_prob,model.predict(elpv_test_imgs))
    Scores.loc['elpv2',OP_NAMES[i]]=score.result().numpy()

    ELPVpred[f'{OP_NAMES[i]}2'] = list(model.predict(elpv_test_imgs))
    PVELXPred[f'{OP_NAMES[i]}2'] = list(model.predict(pvel_test))
    del model,fc
# ------------------------------ ELPV CWCCLoss -----------------------------------

    conv = transfer_net((SIZE,SIZE,3))
    if augmentation:
        datagen = AugmentationBlock((SIZE,SIZE,3), augmentation, make_model=True)
        conv = Model(datagen.input,conv(datagen.output))

    fc = DefectRatingCls(conv.output.shape[1:], ELPV_numclasses)

    start = time.time()
    model = modeling_unit(aug_imgs, pd.get_dummies(aug_proba), conv, fc, CWCCLoss,
                            ['accuracy', F1Score(num_classes = ELPV_numclasses,average='macro'), CostMetric],val_dat = (elpv_val_imgs,elpv_val_prob), 
                            CB=[mcc4,reduce_lr], save_history=ELPV_hist, name=f'CWCC-{OP_NAMES[i]}',
                            BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS[3], LR=LR[3], clswght=CLSWGHTS)


    Train_times.loc['elpv_CWCC', OP_NAMES[i]] = (time.time() - start)/60

    model.load_weights(PATH_CH4)

    pred = model.predict(elpv_test_imgs)
    score = F1Score(num_classes = ELPV_numclasses,average = 'macro')
    score.update_state(elpv_test_prob, pred)
    Scores.loc['elpv_CWCC',OP_NAMES[i]]=score.result().numpy()

    ELPVpred[f'CWCC_{OP_NAMES[i]}'] = list(pred)
    PVELXPred[f'CWCC_{OP_NAMES[i]}'] = list(model.predict(pvel_test))



print(Scores)
ELPV_hist.to_csv('results/elpv_hist.csv',sep=';')
PVEL_hist.to_csv('results/pvel_hist.csv',sep=';')

ELPVpred.to_csv('results/elpv_pred.csv',sep=';')
PVELpred.to_csv('results/pvel_pred.csv',sep=';')
ELPVXPred.to_csv('results/elpv_Xpred.csv',sep=';')
PVELXPred.to_csv('results/pvel_Xpred.csv',sep=';')
Scores.to_csv('results/F1_scores.csv',sep=';')
Train_times.to_csv('results/training_times.csv',sep=';')
