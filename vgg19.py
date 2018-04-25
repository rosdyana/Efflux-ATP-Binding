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
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
import argparse

import time
from datetime import timedelta, datetime


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


def build_model(input_shape, classes):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten(name='flatten')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='vgg19')

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=20)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=1)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-p', '--dataset_prefix',
                        help='Dataset prefix', default="20")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="hasilnyavgg19.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (17, 20, channel)
    # bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

    print("loading dataset")
    dataset_prefix = args.dataset_prefix
    dataset_training = "similar{}training.csv".format(dataset_prefix)
    dataset_validation = "similar{}validation.csv".format(dataset_prefix)
    dataset_independent = "similar{}independent.csv".format(dataset_prefix)
    windowsize = 17
    X_train, Y_train = dataPreprocessing(dataset_training, windowsize)
    X_val, Y_val = dataPreprocessing(dataset_validation, windowsize)
    X_ind, Y_ind = dataPreprocessing(dataset_independent, windowsize)

    Y_train = labelToOneHot(Y_train)
    Y_val = labelToOneHot(Y_val)
    Y_ind = labelToOneHot(Y_ind)
    nb_classes = 2

    model = build_model(SHAPE, nb_classes)

    model.compile(optimizer=Adam(lr=1.0e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(X_val, Y_val))

    # Save Model or creates a HDF5 file
    model.save('{}vgg19_model.h5'.format(time.monotonic()), overwrite=True)
    # del model  # deletes the existing model
    predicted = model.predict(X_ind)
    y_pred = np.argmax(predicted, axis=1)
    Y_ind = np.argmax(Y_ind, axis=1)
    cm = confusion_matrix(Y_ind, y_pred)
    report = classification_report(Y_ind, y_pred)
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
    _fpr, _tpr, _threshold = roc_curve(Y_ind, y_pred)
    AUC = auc(_fpr, _tpr)
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

    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write("{}\n".format(datetime.now))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('AUC: {}\n'.format(AUC))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
