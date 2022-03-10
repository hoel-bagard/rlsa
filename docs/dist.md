## Local install
If you want to install the package locally to test changes, you can use (from virtualenv):
```
python setup.py install
```
## PyPI
### Source Distribution
#### Create the tar file
```
python setup.py sdist
```

### Built Distributions
#### Create the docker
Assuming you are in the repo's folder, enter the manylinux docker with:
```
docker run -ti -v $(pwd)/:/rlsa/ --name rlsa_build --rm quay.io/pypa/manylinux_2_24_x86_64 bash
```
Note: you might want to use `umask 000` (just in case).

#### Create the wheels
From inside the docker use:
```
mkdir rlsa/output
/opt/python/cp310-cp310/bin/python -m pip wheel /rlsa/ -w /rlsa/output
auditwheel repair rlsa/output/rlsa*whl -w rlsa/output
```
The resulting wheel will be `/rlsa/output/rlsa-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl` (with the correct version name).\
Move it to the dist folder.

You can then repeat the operation for each target version of python (changing `cp310-cp310` to `cp39-cp39`, etc..).

### Upload the results
```
pip install --upgrade build twine
```
Note: Both packages are on the arch repo.

#### Test using testpypi
Upload with:
```
twine upload --repository testpypi dist/*
```
And then pip install it (in a venv) with:
```
pip install --index-url https://test.pypi.org/simple/ rlsa
```

#### Upload to PyPI
If the step above worked, then upload it to the real index with:
```
twine upload dist/*
```
The package should then be pip installable.
