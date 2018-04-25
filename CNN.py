import pandas as pd
import numpy as np
# Importing the Keras libraries and packages
import keras
print("keras version : ", keras.__version__)
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D  # pooling step, add pooling layers
# flatten, convert feature maps into large feature vector
from keras.layers import Flatten
from keras.layers import Dropout
# to add the fully connected layers and classic artificial neural network
from keras.layers import Dense
from keras.layers import Convolution2D
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.utils import class_weight
from keras import backend as K
import timeit
from datetime import timedelta, datetime
from keras.models import model_from_json
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
import sys
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
import keras.backend as K


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


def build_and_compile_the_model_for_training(windowsize):
    # Initialising the CNN
    classifier = Sequential()

    # Step 1
    classifier.add(Convolution2D(32, 3, 3, init='glorot_uniform', border_mode='same', input_shape=(
        windowsize, 20, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 1
    classifier.add(Convolution2D(48, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Step 1
    classifier.add(Convolution2D(64, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 1
    classifier.add(Convolution2D(96, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection

    classifier.add(Dense(output_dim=256, activation='relu'))
    # Dropout
    classifier.add(Dropout(0.5))

    classifier.add(Dense(output_dim=2, activation='softmax'))

    classifier.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    return classifier


def convert2OneHot(label):
    tail = (label + 1) % 2
    return np.array([label, tail])


def classificationPerformanceByThreshold(threshold, y_pred, y_true):
    Y_pred = np.empty_like(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i][0] >= threshold:
            Y_pred[i] = np.array([1, 0])
        else:
            Y_pred[i] = np.array([0, 1])

    Y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true, Y_pred)

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
    _fpr, _tpr, _threshold = roc_curve(y_true, Y_pred)
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

    return accuracy, specitivity, sensitivity, mcc, AUC, TPR, FPR, tp, tn, fp, fn


def train_get_result_with_threshold_for_all_fold(dataset_training, dataset_independent, dataset_validation, nb_epoch, batch_size, windowsize, finalResultFile):

    XTrain, YTrain = dataPreprocessing(dataset_training, windowsize)
    XInd, YInd = dataPreprocessing(dataset_independent, windowsize)
    XVal, YVal = dataPreprocessing(dataset_validation, windowsize)

    YTrain = labelToOneHot(YTrain)
    YInd = labelToOneHot(YInd)
    YVal = labelToOneHot(YVal)

    classifier = build_and_compile_the_model_for_training(windowsize)

    classifier.fit(XTrain, YTrain, batch_size=batch_size,
                   nb_epoch=nb_epoch, verbose=2, validation_data=(XVal, YVal))

    classifier.save('CNN' + str(windowsize) + '.h5')

    y_pred = classifier.predict(XInd)

    del classifier

    f_output = open(finalResultFile, "a")
    threshold = [0.149, 0.5]
    # while threshold < 1:
    for i in threshold:
        accuracy, specitivity, sensitivity, mcc, AUC, TPR, FPR, tp, tn, fp, fn = classificationPerformanceByThreshold(
            i, y_pred, YInd)
        f_output.write('=======\n')
        f_output.write('threshold: {}\n'.format(i))
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
        f_output.write('=======\n')
    f_output.close()


start = timeit.default_timer()
print("Start at " + str(start))

finalResultFile = "CNNresult.csv"
windowsize = 17
nb_epoch = 100
batch_size = 64
dataset_prefix = sys.argv[3]
dataset_training = "similar{}training.csv".format(dataset_prefix)
dataset_independent = "similar{}independent.csv".format(dataset_prefix)
dataset_validation = "similar{}validation.csv".format(dataset_prefix)
train_get_result_with_threshold_for_all_fold(
    dataset_training, dataset_independent, dataset_validation, nb_epoch, batch_size, windowsize, finalResultFile)

stop = timeit.default_timer()
print("\nStop at " + str(stop))
print("\nDuration: " + str(stop - start))
print("\n------------------------------------------------------\n")
