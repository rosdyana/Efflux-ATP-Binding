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
# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras import backend as K
import timeit
from keras.models import model_from_json
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
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


# xem ra cau hinh cu the cho model, CV va IND deu phai dung cung 1 model
def build_and_compile_the_model_for_training(windowsize):
    # Initialising the CNN
    classifier = Sequential()

    # Step 1
    classifier.add(Convolution2D(32, 3, 3, init='glorot_uniform', border_mode='same', input_shape=(
        windowsize, 20, 1), activation='relu'))  # version 1.2.2: dung 3,3 thay cho (3,3)
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 1
    classifier.add(Convolution2D(48, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))  # version 1.2.2: dung 3,3 thay cho (3,3)
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Step 1
    classifier.add(Convolution2D(64, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))  # version 1.2.2: dung 3,3 thay cho (3,3)
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 1
    classifier.add(Convolution2D(96, 3, 3, init='glorot_uniform', border_mode='same',
                                 activation='relu'))  # version 1.2.2: dung 3,3 thay cho (3,3)
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    # version 1.2.2: dung output_dim thay do units
    classifier.add(Dense(output_dim=256, activation='relu'))
    # Dropout
    classifier.add(Dropout(0.5))

    classifier.add(Dense(output_dim=2, activation='softmax'))

    # Compiling the CNN
    # Metric values are recorded at the end of each epoch on the training dataset.
    # If a validation dataset is also provided, then the metric recorded is also calculated for the validation dataset.
    classifier.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    return classifier


def convert2OneHot(label):
    tail = (label + 1) % 2
    return np.array([label, tail])


def classificationPerformanceByThreshold(threshold, y_pred, y_true):
    # Y_pred trả về từ function model.predict, dù hàm activation là softmax hay sigmoid thì cũng sẽ có dạng
    # 1 list các probability
    # dùng threshold để quy kết quả về class tương ứng, sau đó đem so với Y_true (là các onehot vector)

    # chú ý cái xử lý, phải copy thằng y_pred ra chứ
    Y_pred = np.empty_like(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i][0] >= threshold:
            Y_pred[i] = np.array([1, 0])  # gán bằng class pos
        else:
            Y_pred[i] = np.array([0, 1])  # gán bằng class neg

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
    # print("tp ", float(tp))
    # print("fn ", float(fn))
    # print("tn ", float(tn))
    # print("fp ", float(fp))

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
    # note: best performance so far by others:  86.13    88.37    87.25    0.75

    return accuracy, specitivity, sensitivity, mcc


# bo X_train, y_train vao day (tuong ung X_test, y_test)
def train_get_result_with_threshold_for_all_fold(dataset, nb_epoch, batch_size, windowsize, finalResultFile):

    # XTrain, YTrain = dataPreprocessing(trainFile, windowsize)
    # YTrain = labelToOneHot(YTrain)

    # XTest, YTest = dataPreprocessing(testFile, windowsize)
    # YTest = labelToOneHot(YTest)

    X, y = dataPreprocessing(dataset, windowsize)

    XTrain, XTest, YTrain, YTest = train_test_split(
        X, y, test_size=0.2, random_state=42)
    YTrain = labelToOneHot(YTrain)
    YTest = labelToOneHot(YTest)

    # print("\nSau khi da dung labelToOneHot")
    # print("\nXTrain shape ", XTrain.shape)
    # print("\nYTrain shape ", YTrain.shape)

    classifier = build_and_compile_the_model_for_training(windowsize)
    # fit the model
    # version 1.2.2: dung nb_epoch thay cho epochs
    classifier.fit(XTrain, YTrain, batch_size=batch_size,
                   nb_epoch=nb_epoch, verbose=2)

    # save model
    classifier.save('20171110.cnn47.Train_Transport_Without_Electron_10_Fold' +
                    str(windowsize) + '.h5')  # creates a HDF5 file 'my_model.h5'

    # lay ket qua predict
    y_pred = classifier.predict(XTest)

    del classifier  # deletes the existing model after it is used to predict

    f2 = open(finalResultFile, "a")
    threshold = [0.149, 0.5]
    # while threshold < 1:
    for i in threshold:
        accuracy, specitivity, sensitivity, mcc = classificationPerformanceByThreshold(
            i, y_pred, YTest)
        f2.write(str(i) + ", " + str(accuracy) + ", " +
                 str(specitivity) + ", " + str(sensitivity) + ", " + str(mcc) + "\n")
    f2.close()


start = timeit.default_timer()
print("Start at " + str(start))

# chay tren transport data, lay tat ca du lieu va chia 10 fold tinh lan luot
# windowsize da duoc tinh lai theo dung chuan
finalResultFile = "finalresult.csv"
windowsize = 17
nb_epoch = int(sys.argv[1])
batch_size = int(sys.argv[2])
dataset = "dataset.csv"
train_get_result_with_threshold_for_all_fold(
    dataset, nb_epoch, batch_size, windowsize, finalResultFile)
# for split in range(1,11):
#     trainFile="data/Transport/sorted-normalized-dataset-csv-single based-without electron/balancedDataset19_"+str(split)+".csv"
#     train_get_result_with_threshold_for_all_fold(split,trainFile,testFile,nb_epoch,windowsize,finalResultFile)

stop = timeit.default_timer()
print("\nStop at " + str(stop))
print("\nDuration: " + str(stop - start))
print("\n------------------------------------------------------\n")
