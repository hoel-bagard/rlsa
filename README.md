# RLSA
C implementation of [RLSA](https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf) for use in python.

## Usage
### Requirements
- Python: 3.8+

### Install
Install with:
```
pip install rlsa
```

### Usage
The main function is `rlsa`.\
It takes as input a black and white image (as a uint8 numpy array), and the hvs, vsv and (optionally) ahvs values.
The function returns a new black and white image, leaving the original one intact.

You can also import the `rlsa_horizontal` and `rlsa_vertical` functions to apply only one of the RLSA components.

### Usage example
A full example would be:
```
import cv2
from rlsa import rlsa

img = cv2.imread("assets/rlsa_test_image.jpg", cv2.IMREAD_GRAYSCALE)
_, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

hsv = vsv = 25
out_img = rlsa(binary_img, hsv, vsv, hsv//10)
```

With a similar setup, the other functions can be used like this:
```
out_img = rlsa_horizontal(binary_img, hsv)
out_img = rlsa_vertical(binary_img, vsv)
```

### Results
| Input image | After RLSA |
|    :---:      |     :---:     |
| ![Input](/assets/rlsa_test_image.jpg?raw "Output sample") | ![Output](/assets/rlsa_out.jpg?raw "Output sample") |

| Horizontal only | Vertical only |
|    :---:      |     :---:     |
| ![Horizontal](/assets/rlsa_out_hor_only.jpg?raw "Horizontal output sample") | ![Vertical](/assets/rlsa_out_vert_only.jpg?raw "Vertical output sample") |


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
