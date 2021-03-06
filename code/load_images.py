import numpy as np
import cv2 as cv

from skimage.io import imread_collection, ImageCollection

import  os


########################################################################################################################

def load_data(path, img_size, flag=0):
    x = imread_collection(os.path.join(path, '*.png'))

    if flag == 1:
        data = np.zeros([len(x.files), img_size, img_size, 3], dtype='uint8')
    else:
        data = np.zeros([len(x.files), img_size, img_size], dtype='uint8')
    for i, fname in enumerate(x.files):
        data[i] = cv.imread(fname, flags=flag)

    return data


def load_images(img_size,gray=0):
    img_size=img_size
    print('Loading data...')

    if gray:
        x_train_path = 'augmented_OD\\patches\\train\\images_gray'
        y_train_path = 'augmented_OD\\patches\\train\\OD_gray'

        X_train = load_data(x_train_path, img_size=img_size, flag=0)

    else:
        x_train_path = 'augmented_OD\\patches\\train\\images'
        y_train_path = 'augmented_OD\\patches\\train\\OD'

        X_train = load_data(x_train_path, img_size=img_size, flag=1)

    # pre-process X
    X_train = X_train / 255.
    y_train = load_data(y_train_path, img_size=img_size)
    #y_train = y_train / 255.
    # pre-process Y
    print(y_train[0].max())
    _, y_train = cv.threshold(y_train, 0.5, 1, cv.THRESH_BINARY)



    return X_train, y_train


def load_images_test(img_size):
    img_size=img_size
    print('Loading data...')
    x_test_path='Dataset_patches\\test\\images'
    y_test_path='Dataset_patches\\test\\OD'

    # test dataset
    X_test=load_data(x_test_path,img_size=img_size)
    # pre-process X
    X_test=X_test/255.
    y_test=load_data(y_test_path,img_size=img_size)
    # pre-process Y
    _,y_test=cv.threshold(y_test,0.5,1,cv.THRESH_BINARY)

    return X_test, y_test