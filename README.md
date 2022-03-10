# RLSA
C implementation of [RLSA](https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf) for use in python.

## Usage
Install it with:
```
pip install rlsa
```
Prerequisites
- python3.5+
- Image must be a binary ndarray(255's/1's/0's)
- Must pass a predefined limit, a certain integer "value"


## Included scripts

A few scripts are included in the tests folder. One is a python implementation of rlsa, serving as reference. The other two compare the result and speed of the implementations. 

### Test
```
python -m tests.test assets/rlsa_test_image.jpg
```

### Benchmark
```
python -m tests.benchmark assets/rlsa_test_image.jpg
```
--> C version is around 400 times faster than the naive python one.



## Misc
Note: the numpy tutorial/doc [here](https://numpy.org/doc/stable/user/c-info.how-to-extend.html) is (it seems) outdated (be carefull when using it).
