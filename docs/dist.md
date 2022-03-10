
First create a virtualenv and enter it. Then:
```
python setup.py install
```

## PyPI steps


docker run -ti -v $(pwd)/:/rlsa/ --name rlsa_build --rm quay.io/pypa/manylinux_2_24_x86_64 bash
```
mkdir rlsa/output
/opt/python/cp310-cp310/bin/python -m pip wheel /rlsa/ -w /rlsa/output
auditwheel repair rlsa/output/rlsa*whl -w rlsa/output
```

/rlsa/output/rlsa-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl

/opt/python/cp38-cp38/bin/python -m pip wheel /rlsa/ -w /rlsa/output



### Upload the results
```
pip install --upgrade build twine
```
Note: twine is also on the arch repo.





#### Test using testpypi
Upload with:
```
twine upload --repository testpypi dist/*
```
And then pip install it (in a venv) with:
```
pip install --index-url https://test.pypi.org/simple/ rlsa
```
