import numpy as np
import setuptools


rlsa_module = setuptools.Extension("rlsa", sources=["rlsa/rlsa.c"], include_dirs=[np.get_include()])


setuptools.setup(
    name="rlsa",
    version="0.0.1",
    description="Run Length Smoothing Algorithm",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url="https://github.com/hoel-bagard/rlsa",
    author="Hoel Bagard",
    classifiers=["Programming Language :: Python :: 3"],
    keywords="rlsa",
    zip_safe=False,
    install_requires=["numpy"],
    ext_modules=[rlsa_module],
    packages=["rlsa"],
    include_package_data=True,
    python_requires=">=3.8",
)


# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration

#     config = Configuration()
#     config.add_extension("rlsa", ["rlsa/rlsa.c"])

#     return config


# if __name__ == "__main__":
#     from numpy.distutils.core import setup
#     setup(configuration=configuration)
