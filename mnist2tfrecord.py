import sys
import os
import numpy as np
import tensorflow as tf
from progressbar import ProgressBar

flags = tf.flags
flags.DEFINE_boolean('train', True, '')
flags.DEFINE_string('path', 'mnist', '')
flags.DEFINE_boolean('validate', True, '')
flags.DEFINE_string('recordname', '', '')
flags.DEFINE_boolean('make', True, '')
cfg = tf.flags.FLAGS


def mnist2tfrd(path, train, recordname):
    # path = 'mnist'
    if train:
        file_pre = 'train'
        num_data = 60000
    else:
        file_pre = 't10k'
        num_data = 10000
        
    if recordname == '':
        recordname = file_pre + '.tfrd'

    fd = open(os.path.join(path, '%s-images-idx3-ubyte' % file_pre))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trainX = loaded[16:].reshape((num_data, 28, 28, 1)).astype(np.int64)

    fd = open(os.path.join(path, '%s-labels-idx1-ubyte' % file_pre))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trainY = loaded[8:].reshape((num_data)).astype(np.int64)

    # print(trainX.shape, trainY.shape)
    # print(trainX[0])

    # make train.tfr
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.ByteList(value=value))

    # setname = 'mnist'
    # recordname = 'original_train.tfrd'
    print('making %s record...' % file_pre)

    writer = tf.python_io.TFRecordWriter(os.path.join(path, recordname))
    pbar = ProgressBar().start()

    for img,label, i in zip(trainX, trainY, range(num_data)):
        # img is in np.int64
        img = np.reshape(img, [-1])
        label_digit = np.zeros([10], dtype=np.int64)
        label_digit[label] = 1

        feature = dict()

        feature['image'] = _int64_feature(img)
        feature['label_digit'] = _int64_feature(label_digit)
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

        pbar.update(i / num_data * 100)

        # if i % 100 == 0:
            # pbar.update(i / num_data * 100)
    writer.close()
    print('%s is complete' % recordname)

    info = dict()
    info['image'] = (28, 28, 1)
    info['label_digit']  = (10)
    import json
    with open('original.json', 'w') as f:
        json.dump(info, f)

if cfg.make:
    mnist2tfrd(cfg.path, cfg.train, cfg.recordname)

def test_tfrd(path, recordname):
    def _parse_function(example_proto):
        features = {'image': tf.FixedLenFeature((28,28,1), tf.int64),
                    'label_digit': tf.FixedLenFeature((10), tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        print(parsed_features)
        for i in parsed_features:
            parsed_features[i] = tf.cast(parsed_features[i], tf.int16)
        print(parsed_features)
        # parsed_features = tf.cast(parsed_features, tf.int32)
        # out_features = (tf.cast(i, tf.int32) for i in parsed_features)
        return parsed_features

    recordname = os.path.join(path, recordname)
    dataset = tf.data.TFRecordDataset(recordname)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        p = sess.run(iterator.get_next())
        print(p)

if cfg.validate:
    test_tfrd(cfg.path, cfg.recordname)
