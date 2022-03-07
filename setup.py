from distutils.core import Extension, setup

rlsa_module = Extension("rlsa", sources=["rlsa/rlsa.c"])

setup(name="rlsa",
      version="0.1",
      description="RLSA package",
      ext_modules=[rlsa_module])
