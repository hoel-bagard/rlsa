# RLSA
C implementation of [RLSA](https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf) for use in python.

## Usage
First create a virtualenv and enter it. Then:
```
python setup.py install && python tests/test.py
```

### Test
```
python -m tests.test assets/rlsa_test_image.jpg
```

### Benchmark
```
python -m tests.benchmark assets/rlsa_test_image.jpg
```
--> C version is basically 800 times faster than the naive python one.

## Misc
Note: the numpy tutorial/doc [here](https://numpy.org/doc/stable/user/c-info.how-to-extend.html) is outdated and seems to be for python2 =(
