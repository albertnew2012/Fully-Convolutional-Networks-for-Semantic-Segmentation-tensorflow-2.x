import os
import random
from itertools import repeat

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model


def load_data(dir_data):
    dir_data = "dataset"
    dir_img_train = dir_data + "/images_prepped_train/"
    dir_label_train = dir_data + "/annotations_prepped_train/"
    dir_img_test = dir_data + "/images_prepped_test/"
    dir_label_test = dir_data + "/annotations_prepped_test/"
    data = []
    for dir in [dir_img_train, dir_img_test, dir_label_train, dir_label_test]:
        files = os.listdir(dir)
        files.sort()
        res = []
        for file in files:
            res.append(cv2.imread(dir + file))
        data.append(np.asarray(res))
    X_train, X_test, y_train, y_test = data
    return X_train, X_test, y_train, y_test


def reduce_resolution(*images, output_height=224, output_width=224):
    res = []
    for image in images:
        image = list(map(cv2.resize, image, repeat((output_height, output_width))))
        res.append(np.asarray(image))
    return res


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 2:
        seg = np.dstack([seg] * 3)
    seg_img = np.zeros_like(seg).astype('float')
    colors = sns.color_palette("hls", n_classes)
    for c in range(n_classes):
        segc = (seg == c)
        seg_img += segc * (colors[c])
    return seg_img


def display_classes(X_train, y_train):
    i = random.randint(0, X_train.shape[0])
    img = X_train[i]
    seg = y_train[i]
    n_classes = np.unique(seg).size
    seg_color = give_color_to_seg_img(seg, n_classes)
    plt.figure(figsize=(15, 35))
    plt.subplot(121)
    plt.title("original image")
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(seg_color)
    plt.title("segmentation label")


def normalize_img(X_train, X_test):
    X_train = X_train / 127.5 - 1
    X_test = X_test / 127.5 - 1
    return X_train, X_test


def expand_label(n_classes, *label_imgs):
    output = []
    for label_img in label_imgs:
        label = np.zeros((*label_img.shape[:-1], n_classes), dtype=int)
        for i in range(n_classes):
            label[:, :, :, i] = (label_img[:, :, :, 0] == i).astype(int)
        output.append(label)
    return output


def FCN8(nClasses, VGG_Weights, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  ## (None, 112, 112, 64)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)  ## (None, 56, 56, 128)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)  ## (None, 28, 28, 256)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)  ## (None, 7, 7, 512)

    _vgg = Model(img_input, pool5)
    _vgg.load_weights(VGG_Weights)  ## loading VGG weights for the encoder parts of FCN8

    ## 2 times upsamping for pool5 layer
    deconv1 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_1")(pool5)
    sum1 = Add(name="sum1")([deconv1, pool4])

    ## 2 times upsamping for deconv1 layer
    deconv2 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_2")(sum1)
    sum2 = Add(name="sum2")([deconv2, pool3])

    ## 8 times upsamping for deconv1 layer
    deconv2 = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, name="upsampling8x")(sum2)
    output = (Activation('softmax'))(deconv2)
    model = Model(img_input, output)
    return model

def FCN2(nClasses, VGG_Weights, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  ## (None, 112, 112, 64)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)  ## (None, 56, 56, 128)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)  ## (None, 28, 28, 256)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)  ## (None, 7, 7, 512)

    _vgg = Model(img_input, pool5)
    _vgg.load_weights(VGG_Weights)  ## loading VGG weights for the encoder parts of FCN8

    ## 2 times upsamping for pool5 layer
    deconv1 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_1")(pool5)
    sum1 = Add(name="sum1")([deconv1, pool4])

    ## 2 times upsamping for deconv1 layer
    deconv2 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_2")(sum1)
    sum2 = Add(name="sum2")([deconv2, pool3])

    ## 2 times upsamping for deconv1 layer
    deconv3 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_3")(sum2)
    sum3 = Add(name="sum3")([deconv3, pool2])

    ## 2 times upsamping for deconv1 layer
    deconv4 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_4")(sum3)
    sum4 = Add(name="sum4")([deconv4, pool1])

    deconv5 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="upsampling2x_5")(sum4)
    output = (Activation('softmax'))(deconv5)
    model = Model(img_input, output)
    return model


def display_loss(history):
    plt.figure(figsize=(30, 10))
    plt.subplot(121)
    for key in ['loss', 'val_loss']:
        plt.plot(history.history[key], label=key)
    plt.legend()
    plt.subplot(122)
    for key in ['accuracy', 'val_accuracy']:
        plt.plot(history.history[key], label=key)
    plt.legend()
    plt.show()
    plt.savefig("result/learning_curve")


def visualize_pred(X_test, y_testi):
    n_classes = 10
    plt.figure(figsize=(10, 30))
    for i, v in enumerate(random.sample(range(X_test.shape[0]), k=3)):
        img = (X_test[v] + 1) * (255.0 / 2)
        seg = y_predi[v]
        segtest = y_testi[v]

        plt.subplot(3, 3, 1 + 3 * i)
        plt.imshow(img / 255.0)
        plt.title("original")

        plt.subplot(3, 3, 2 + 3 * i)
        plt.imshow(give_color_to_seg_img(seg, n_classes))
        plt.title("predicted class")

        plt.subplot(3, 3, 3 + 3 * i)
        plt.imshow(give_color_to_seg_img(segtest, n_classes))
        plt.title("true class")
        plt.show()
    plt.savefig("result/predictions")


def IoU(y_testi, y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    IoUs = []
    n_classes = int(np.max(y_testi)) + 1
    for c in range(n_classes):
        TP = np.sum((y_testi == c) & (y_predi == c))
        FP = np.sum((y_testi != c) & (y_predi == c))
        FN = np.sum((y_testi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("____________________________________________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


if __name__ == "__main__":
    ## load data
    dir_data = "dataset"
    X_train, X_test, y_train, y_test = load_data(dir_data)
    display_classes(X_train, y_train)
    classes = set(np.unique(y_train).tolist() + np.unique(y_test).tolist())
    n_classes = len(classes)
    print(f"Total number of segmentation classes = {n_classes}\nclass IDs are {classes}")

    ## preprocessing
    X_train, X_test, y_train, y_test = reduce_resolution(X_train, X_test, y_train, y_test)
    display_classes(X_train, y_train)
    X_train, X_test = normalize_img(X_train, X_test)
    y_train, y_test = expand_label(n_classes, y_train, y_test)

    ## create a model
    if not os.path.isfile("VGG_Weights.h5"):
        model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model_vgg16.save_weights("VGG_Weights.h5")
    model = FCN8(n_classes, "VGG_Weights.h5")
    sgd = SGD(lr=1E-2, decay=5 ** (-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=45, verbose=2)
    model.summary()
    plot_model(model)
    model.save("FCN8s")

    ## Plot loss and acc
    display_loss(history)

    ## load model
    # model = load_model("FCN8s")
    ## make prediction
    y_pred = model.predict(X_test)
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test, axis=3)
    print(y_testi.shape, y_predi.shape)

    ## Visualize the model performance
    visualize_pred(X_test, y_testi)

    ## Calculate intersection over union for each segmentation class
    IoU(y_testi, y_predi)
