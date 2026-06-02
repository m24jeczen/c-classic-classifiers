import setuptools
import numpy 

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
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"]
        )
    ]
)