import tensorflow as tf  # uncomment this for using GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# maximun alloc gpu50% of MEM
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# allocate dynamically
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


import math
import json
import sys

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
import argparse

import time
from datetime import timedelta


def changeLabel(label):
    for i in range(len(label)):
        if label[i] == 2:
            label[i] = 0


def labelToOneHot(label):  # 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis=1)
    label[:, 0] = label[:, 0] == 0
    return label


def dataPreprocessing(dataFile, windowsize):
    # data pre-processing
    data = pd.read_csv(dataFile, header=None)
    X = data.iloc[:, 0:windowsize * 20].values
    y = data.iloc[:, windowsize * 20].values
    X = X.reshape(len(X), windowsize, 20, 1)
    changeLabel(y)
    return X, y


modelinput = sys.argv[1]
outputresult = sys.argv[2]
dataset_test = sys.argv[3]

print("loading dataset")
windowsize = 17
X_test, Y_test = dataPreprocessing(dataset_test, windowsize)
Y_test = labelToOneHot(Y_test)

model = load_model(modelinput)

predicted = model.predict(X_test)
y_pred = np.argmax(predicted, axis=1)
Y_test = np.argmax(Y_test, axis=1)
cm = confusion_matrix(Y_test, y_pred)
report = classification_report(Y_test, y_pred)
tn = cm[0][0]
fn = cm[1][0]
tp = cm[1][1]
fp = cm[0][1]
if tp == 0:
    tp = 1
if tn == 0:
    tn = 1
if fp == 0:
    fp = 1
if fn == 0:
    fn = 1
TPR = float(tp) / (float(tp) + float(fn))
FPR = float(fp) / (float(fp) + float(tn))
accuracy = round((float(tp) + float(tn)) / (float(tp) +
                                            float(fp) + float(fn) + float(tn)), 3)
specitivity = round(float(tn) / (float(tn) + float(fp)), 3)
sensitivity = round(float(tp) / (float(tp) + float(fn)), 3)
mcc = round((float(tp) * float(tn) - float(fp) * float(fn)) / math.sqrt(
    (float(tp) + float(fp))
    * (float(tp) + float(fn))
    * (float(tn) + float(fp))
    * (float(tn) + float(fn))
), 3)

f_output = open(outputresult, 'a')
f_output.write('=======\n')
f_output.write('TN: {}\n'.format(tn))
f_output.write('FN: {}\n'.format(fn))
f_output.write('TP: {}\n'.format(tp))
f_output.write('FP: {}\n'.format(fp))
f_output.write('TPR: {}\n'.format(TPR))
f_output.write('FPR: {}\n'.format(FPR))
f_output.write('accuracy: {}\n'.format(accuracy))
f_output.write('specitivity: {}\n'.format(specitivity))
f_output.write("sensitivity : {}\n".format(sensitivity))
f_output.write("mcc : {}\n".format(mcc))
f_output.write("{}".format(report))
f_output.write('=======\n')
f_output.close()
print("DONE")
