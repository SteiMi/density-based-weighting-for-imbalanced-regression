import cv2
import os, sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
try:
    import matlab.engine
    MATLAB_ENGINE = matlab.engine.start_matlab("-nojvm")
except ImportError as err:
    print err

upsample = 2
sub_image_size = 31
stride = 15

chunksize = 200
train_ratio = 1.0

raw_path = os.path.join("data")
train_path = os.path.join(raw_path, "train")
test_path = os.path.join(raw_path, "test")

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_image(img, upsample=upsample, stride=15, imsize=31, is_training=True, w_matlab=False):
    inputs, labels = [], []
    h, w, d = img.shape
    imgsub = img[:upsample*int(h/upsample), :upsample*int(w/upsample)]
    if w_matlab:
        A = matlab.double(imgsub.tolist())
        img_l_matlab = MATLAB_ENGINE.imresize(A, 1./upsample, 'bicubic')
        img_up_matlab = MATLAB_ENGINE.imresize(img_l_matlab, upsample, 'bicubic')
        img_l = np.array(img_l_matlab)
        imgup = np.array(img_up_matlab)
    else:
        img_l = cv2.resize(imgsub, (0,0), fx=1./upsample, fy=1./upsample, interpolation=cv2.INTER_CUBIC)
        imgup = cv2.resize(img_l, (0,0), fx=1.*upsample, fy=1.*upsample, interpolation=cv2.INTER_CUBIC)


    imgsub = imgsub[upsample:-upsample, upsample:-upsample]
    imgup = imgup[upsample:-upsample, upsample:-upsample]
    h, w, d = imgsub.shape # reset with the new dimensions

    if not is_training:
        return imgup[np.newaxis, :, :, np.newaxis], imgsub[np.newaxis]

    for y in np.arange(0, h, stride):
        for x in np.arange(0, w, stride):
            ylow, yhigh = y, y+imsize
            xlow, xhigh = x, x+imsize
            if (xhigh > w) or (yhigh > h):
                continue

            labels += [imgsub[np.newaxis, ylow:yhigh, xlow:xhigh]]
            inputs += [imgup[np.newaxis, ylow:yhigh, xlow:xhigh, np.newaxis]]

    return np.concatenate(inputs, axis=0), np.concatenate(labels, axis=0)

def get_luminance(img):
    coefs = np.array([[0.114], [0.587], [0.299]])
    y = np.squeeze(img.dot(coefs))
    y = np.round((y / 255.) * (235-16) + 16) #matlab returns Y in range [16,235]
    return y

def write_records(inputs, labels, write_dir, name):
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    writer = tf.python_io.TFRecordWriter(os.path.join(write_dir, name + '.tfrecords'))
    num_examples = len(inputs)
    for index in range(num_examples):
        n, height, width, depth = inputs[index].shape
        for j in range(n):
            x_in = inputs[index][j].astype(np.float32).tostring()
            lab = labels[index][j].astype(np.float32).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(lab),
                'image': _bytes_feature(x_in),
                'depth': _int64_feature(depth),
                'height': _int64_feature(height),
                'width': _int64_feature(width)
                }))
            writer.write(example.SerializeToString())
    writer.close()

def build_dataset(filelist, is_training=True):
    file_count = 0
    X, Y = [], []
    tffile_dir = os.path.dirname(filelist[0]) + "_tfrecords_%i" % upsample
    for j, f in enumerate(filelist):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        img = get_luminance(img)
        img = img[:,:,np.newaxis]
        inputs, labels = process_image(img, is_training=is_training, w_matlab=False)
        X.append(inputs)
        Y.append(labels)

        if (j % chunksize == 0) and (j!=0):
            if is_training:
                write_records(X, Y, tffile_dir, 'train_%i' % file_count)
            else:
                write_records(X, Y, tffile_dir, 'test_%i' % file_count)
            print "Training:", is_training, " File number:", file_count
            file_count += 1
            X, Y = [], []

    if is_training:
        write_records(X, Y, tffile_dir, 'train_%i' % file_count)
    else:
        write_records(X, Y, tffile_dir, 'test_%i' % file_count)

if __name__ == '__main__':
    set5_path = os.path.join(test_path, "Set5")
    set14_path = os.path.join(test_path, "Set14")
    train_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(train_path) for f in filenames if
              os.path.splitext(f)[1] == '.bmp']
    set5_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(set5_path) for f in filenames if
              os.path.splitext(f)[1] == '.bmp']
    set14_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(set14_path) for f in filenames if
              os.path.splitext(f)[1] == '.bmp']
    build_dataset(train_files, is_training=True)
    build_dataset(set5_files, is_training=False)
    build_dataset(set14_files, is_training=False)
