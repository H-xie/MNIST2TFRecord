# MNIST2TFRecord

This is a simple way to convert original **mnist** to `TFRecord` for a geater effective.

## Preparation
Some packages is need
```shell
pip3 install tensorflow numpy opencv-python 
```

## Usage
1. Need to download and unzip [mnist](http://yann.lecun.com/exdb/mnist/) by
  ```shell
  python3 download_data.py
  ```
2. Run the script to generate tfrecord.
  ```shell
  python3 mnist2tfrecord.py
  ```
  This contains several default options. `-h` will show all infomation.

## Reference
- `download_data.py` is directly cited from [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)

- [mnist](http://yann.lecun.com/exdb/mnist/) is a popular handwritten digits dataset.
