# RLSA
C implementation of [RLSA](https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf) for use in python.

## Usage
### Requirements
- python version >= 3.8

### Install
Install with:
```
pip install rlsa
```

### Usage
The only function currently exported is `rlsa`.\
It takes as input a black and white image (as a uint8 numpy array), and the hvs and vsv values (for now the ahsv is fixed as `hsv // 10`).\
The function returns a new black and white image, leaving the original one intact.

### Usage example
```
import cv2
from rlsa import rlsa

img = cv2.imread("assets/rlsa_test_image.jpg", cv2.IMREAD_GRAYSCALE)
_, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

hsv = vsv = 25
out_img = rlsa(binary_img, hsv, vsv)
```

### Results
TODO

## Included scripts

A few scripts are included in the tests folder. One is a python implementation of rlsa, serving as reference. The other two compare the result and speed of the implementations.\
To run the those scripts, you need to install opencv.

### Test
```
python -m tests.test assets/rlsa_test_image.jpg
```

### Benchmark
```
python -m tests.benchmark assets/rlsa_test_image.jpg
```
--> C version is around 400 times faster than the naive python one.



#### Misc
Note: the numpy tutorial/doc [here](https://numpy.org/doc/stable/user/c-info.how-to-extend.html) is (it seems) outdated (be carefull when using it).
