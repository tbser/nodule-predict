#!usr/bin/env python
# -*-coding:utf-8-*-

from __future__ import print_function
import numpy

from keras import backend as K

from keras import layers
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD
from keras.metrics import binary_accuracy, binary_crossentropy, mean_absolute_error

import h5py

CUBE_SIZE = 32
LEARN_RATE = 0.001


def read_img(img_path):
    """
    this function returns preprocessed image
    """
    dim_ordering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img, dim_ordering=dim_ordering)

    # decenterize
    img[0, :, :] -= mean[0]
    img[1, :, :] -= mean[1]
    img[2, :, :] -= mean[2]

    # 'RGB'->'BGR'
    if dim_ordering == 'th':
        img = img[::-1, :, :]
    else:
        img = img[:, :, ::-1]

    # expand dim for test
    img = numpy.expand_dims(img, axis=0)
    return img


def identity_block(input_tensor, nb_filter, stage, block, kernel_size=3):
    """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        bn_axis = 4
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = nb_filter     # 512,512,2048

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution3D(nb_filter1, (1, 1, 1), name=conv_name_base + '2a', data_format="channels_last")(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    # 为了使得输入的shape与经过主路的shape一样，在卷积层中设置border_mode=’same’，则卷积层会自动保证输入输出具有相同的shape
    out = Convolution3D(nb_filter2, kernel_size, name=conv_name_base + '2b', padding="same", data_format="channels_last")(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', data_format="channels_last")(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    out = layers.add([out, input_tensor])
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, nb_filter, stage, block, kernel_size=3, strides=(2, 2, 2)):
    """
    conv_block indicate the block that has a conv layer at shortcut
    params:
        input_tensor: input tensor
        nb_filter: list of integers, the nb_filters of 3 conv layer at main path
        stage: integet, current stage number
        block: str like 'a','b'.., current block
        kernel_size: defualt 3, the kernel size of middle conv layer at main path

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        bn_axis = 4
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 只有stage2的第一个卷积是strides=(1, 1, 1),其他都是默认(2, 2, 2)
    out = Convolution3D(nb_filter1, (1, 1, 1), name=conv_name_base + '2a', strides=strides, data_format="channels_last")(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution3D(nb_filter2, kernel_size, name=conv_name_base + '2b', padding='same', data_format="channels_last")(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', data_format="channels_last")(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    # 为了使得两个张量可以相加，shortcut这条路的滤波器数目一定和主路最后一个卷积模块的滤波器数目是一样的。
    shortcut = Convolution3D(nb_filter3, (1, 1, 1), name=conv_name_base + '1', strides=strides, data_format="channels_last")(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    out = layers.add([out, shortcut])
    out = Activation('relu')(out)
    return out


def get_resnet50(input_shape, load_weight_path=None):
    """
    this function return the resnet50 model
    you should load pretrained weights if you want to use this model directly
    Note that since the pretrained weights were converted from caffemodel
    so the order of channels for input image should be 'BGR' (the channel order of caffe)
    """

    if K.image_dim_ordering() == 'tf':
        # input_shape = (224, 224, 224, 3)
        # input_shape = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        bn_axis = 4
    else:
        # input_shape = (3, 224, 224, 224)
        # input_shape = (1, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        bn_axis = 1

    inp = Input(shape=input_shape)
    out = ZeroPadding3D((3, 3, 3))(inp)
    out = Convolution3D(64, (7, 7, 7), name='conv1', strides=(2, 2, 2), data_format="channels_last")(out)
    out = BatchNormalization(axis=bn_axis, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), data_format="channels_last")(out)

    out = conv_block(out, [64, 64, 256], 2, 'a', strides=(1, 1, 1))
    out = identity_block(out, [64, 64, 256], 2, 'b')
    out = identity_block(out, [64, 64, 256], 2, 'c')

    out = conv_block(out, [128, 128, 512], 3, 'a')
    out = identity_block(out, [128, 128, 512], 3, 'b')
    out = identity_block(out, [128, 128, 512], 3, 'c')
    out = identity_block(out, [128, 128, 512], 3, 'd')

    out = conv_block(out, [256, 256, 1024], 4, 'a')
    out = identity_block(out, [256, 256, 1024], 4, 'b')
    out = identity_block(out, [256, 256, 1024], 4, 'c')
    out = identity_block(out, [256, 256, 1024], 4, 'd')
    out = identity_block(out, [256, 256, 1024], 4, 'e')
    out = identity_block(out, [256, 256, 1024], 4, 'f')

    out = conv_block(out, [512, 512, 2048], 5, 'a')
    out = identity_block(out, [512, 512, 2048], 5, 'b')
    out = identity_block(out, [512, 512, 2048], 5, 'c')

    out = AveragePooling3D(1, (7, 7, 7), data_format="channels_last")(out)
    # out = Flatten()(out)
    # out = Dense(1000, activation='softmax', name='fc1000')(out)

    out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid", name="out_class_last")(out)
    out_class = Flatten(name="out_class")(out_class)

    out_malignancy = Convolution3D(1, 1, 1, 1, activation=None, name="out_malignancy_last")(out)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    model = Model(input=inp, output=[out_class, out_malignancy])
    
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)    

    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True),
                  loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error},
                  metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    return model


# def load_weights(model, weights_path):
#     """
#     This function load the pretrained weights to the model
#     """
#     f = h5py.File(weights_path, 'r')
#     for layer in model.layers:
#         if layer.name[:3] == 'res':
#             layer.set_weights([f[layer.name]['weights'][:], f[layer.name]['bias'][:]])
#         elif layer.name[:2] == 'bn':
#             scale_name = 'scale'+layer.name[2:]
#             weights = []
#             # 注意这里的权重组织顺序
#             weights.append(f[scale_name]['weights'][:])
#             weights.append(f[scale_name]['bias'][:])
#             weights.append(f[layer.name]['weights'][:])
#             weights.append(f[layer.name]['bias'][:])
#
#             layer.set_weights(weights)
#     model.get_layer('conv1').set_weights([f['conv1']['weights'][:], f['conv1']['bias'][:]])
#     model.get_layer('fc1000').set_weights([f['fc1000']['weights'][:].T, f['fc1000']['bias'][:]])
#     return model


# if __name__ == '__main__':
#     K.set_image_dim_ordering('tf')
#     weights_file = K.image_dim_ordering() + '_dim_ordering_resnet50.h5'
#     resnet_model = get_resnet50()
#     resnet_model.load_weights(weights_file)
#     test_img1 = read_img('cat.jpg')
#     test_img2 = read_img('airplane.jpg')
#     # you may download synset_words from address given at the beginning of this file
#     class_table = open('synset_words', 'r')
#     lines = class_table.readlines()
#     print("result for test 1 is")
#     print(lines[numpy.argmax(resnet_model.predict(test_img1)[0])])
#     print("result for test 2 is")
#     print(lines[numpy.argmax(resnet_model.predict(test_img2)[0])])
#     class_table.close()
