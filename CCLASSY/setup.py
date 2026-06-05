import setuptools
import numpy 
import sys
import os

extra_compile_args = []
extra_link_args = []

if sys.platform == 'darwin':
    extra_compile_args = [
        '-Xpreprocessor', '-fopenmp',
        '-I/opt/homebrew/opt/libomp/include'
    ]
    extra_link_args = [
        '-L/opt/homebrew/opt/libomp/lib',
        '-lomp'
    ]
    
elif sys.platform == 'linux':
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

setuptools.setup(
    name="CCLASSY",
    packages=setuptools.find_packages(),
    ext_modules=[
        setuptools.Extension(
            "CCLASSY.cmodule",
            sources=[
                "src/cmodule.c",
                "src/common.c",
                "src/knn.c",
                "src/naive_bayes.c",
                "src/logistic_regression.c",
                "src/svm.c"
            ],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ]
)