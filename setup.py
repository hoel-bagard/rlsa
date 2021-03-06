import numpy as np
import setuptools

rlsa_module = setuptools.Extension("rlsa", sources=["rlsa/rlsa.c"], include_dirs=[np.get_include()])


setuptools.setup(
    name="rlsa",
    version="0.0.2",
    description="Run Length Smoothing Algorithm",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url="https://github.com/hoel-bagard/rlsa",
    author="Hoel Bagard",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
    keywords="rlsa",
    zip_safe=False,
    install_requires=["numpy"],
    ext_modules=[rlsa_module],
    packages=["rlsa"],
    include_package_data=True,
    python_requires=">=3.8",
)
