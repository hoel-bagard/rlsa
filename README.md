# RLSA
C implementation of [RLSA](https://users.iit.demokritos.gr/~bgat/RLSA_values.pdf) for use in python.

## Usage
First create a virtualenv and enter it. Then:
```
python setup.py install
```

### Test
```
python -m tests.test assets/rlsa_test_image.jpg
```

### Benchmark
```
python -m tests.benchmark assets/rlsa_test_image.jpg
```
--> C version is around 400 times faster than the naive python one.


## Pypi steps
pip install --upgrade build twine
python -m build
twine upload --repository testpypi dist/*


## Misc
Note: the numpy tutorial/doc [here](https://numpy.org/doc/stable/user/c-info.how-to-extend.html) is (it seems)outdated  =(
