# /**
#  * @author [Rosdyana Kusuma]
#  * @email [rosdyana.kusuma@gmail.com]
#  * @create date 2018-04-24 09:25:10
#  * @modify date 2018-04-24 09:25:10
#  * @desc [DenseNet - Keras]
# */

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # uncomment this for using GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=2):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    return model


def DenseNet121(input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=2):
    return DenseNet([6, 12, 24, 16],
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet169(input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=2):
    return DenseNet([6, 12, 32, 32],
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet201(input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=2):
    return DenseNet([6, 12, 48, 32],
                    input_tensor, input_shape,
                    pooling, classes)


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
    parser.add_argument('-l', '--layernumber',
                        help='layernumber', type=str, default="121")
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

    if args.layernumber == '121':
        model = DenseNet121(input_shape=SHAPE)
    elif args.layernumber == '169':
        model = DenseNet169(input_shape=SHAPE)
    elif args.layernumber == '201':
        model = DenseNet201(input_shape=SHAPE)

    model.compile(optimizer=Adam(lr=1.0e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(X_val, Y_val))

    # Save Model or creates a HDF5 file
    model.save('{}vgg16_model.h5'.format(time.monotonic()), overwrite=True)
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
    _fpr, _tpr, _threshold = roc_curve(Y_test, y_pred)
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
    f_output.write('TPR: {}\n'.format(_fpr))
    f_output.write('FPR: {}\n'.format(_tpr))
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
