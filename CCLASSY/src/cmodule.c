#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "common.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

int* knn_predict(
    const double* X_train, const int* y_train,
    size_t n_train, size_t n_features, 
    const double* X_test, size_t n_test, int k);

static PyObject* py_knn_predict(PyObject* self, PyObject* args){
    PyArrayObject *X_train_arr, *y_train_arr, *X_test_arr;
    int k;

    if(!PyArg_ParseTuple(
        args, "O!O!O!i",
        &PyArray_Type, &X_train_arr,
        &PyArray_Type, &y_train_arr,
        &PyArray_Type, &X_test_arr,
        &k
    ))
    return NULL;

    if (PyArray_TYPE(X_train_arr) != NPY_DOUBLE ||
        PyArray_TYPE(X_test_arr) != NPY_DOUBLE ||
        PyArray_TYPE(y_train_arr) != NPY_INT)
            return PyErr_Format(PyExc_TypeError, 
            "X must be float64, y must be int32");

    if (!PyArray_IS_C_CONTIGUOUS(X_train_arr) ||
        !PyArray_IS_C_CONTIGUOUS(X_test_arr) ||
        !PyArray_IS_C_CONTIGUOUS(y_train_arr))
        return PyErr_Format(PyExc_ValueError,
        "arrays must be C-contiguous");

    if (PyArray_NDIM(X_train_arr) != 2 ||
        PyArray_NDIM(X_test_arr) != 2 ||
        PyArray_NDIM(y_train_arr) != 1)
            return PyErr_Format(PyExc_TypeError, 
            "X must be 2D, y must be 1D");

    size_t n_train = (size_t)PyArray_DIM(X_train_arr, 0);
    size_t n_features = (size_t)PyArray_DIM(X_train_arr, 1);
    size_t n_test = (size_t)PyArray_DIM(X_test_arr, 0);

    if (PyArray_DIM(X_test_arr, 1) != (npy_intp)n_features)
        return PyErr_Format(PyExc_ValueError,
        "X_train and X_test must have the same number of features");

    if ((size_t)PyArray_DIM(X_train_arr, 0) != n_train)
        return PyErr_Format(PyExc_ValueError,
        "X_train and y_train must have the same number of samples");

    if (k <= 0 || (size_t)k > n_train)
        return PyErr_Format(PyExc_ValueError,
        "k must be between 1 and n_train");

    const double* X_train = (const double*)PyArray_DATA(X_train_arr);
    const int* y_train  = (const int*)PyArray_DATA(y_train_arr);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);

    int* predictions = knn_predict(
        X_train, y_train, n_train, n_features, X_test, n_test,k);

    if(!predictions)
        return PyErr_Format(PyExc_RuntimeError,
        "knn_predict failed (memory allocation error)");

    npy_intp dims[1] = {(npy_intp)n_test};
    PyObject* result = PyArray_SimpleNewFromData(
        1, dims, NPY_INT, predictions);

    if (!result) {
        free(predictions);
        return PyErr_Format(PyExc_RuntimeError,
        "failed to create output array");
    }

    PyArray_ENABLEFLAGS((PyArrayObject*)result, NPY_ARRAY_OWNDATA);
    return result;
    }
    static PyMethodDef cclassy_methods[] = {
        {"py_knn_predict", py_knn_predict, METH_VARARGS,
        "KNN prediction: py_knn_predict(X_train, y_train, X_test, k)"},
        {NULL, NULL, 0, NULL}
    };

    static struct PyModuleDef cclassy_module = {
        PyModuleDef_HEAD_INIT,
        "cmodule",
        "CCLASSY C extension module",
        -1,
        cclassy_methods
    };

    PyMODINIT_FUNC PyInit_cmodule(void) {
        PyObject* mod  = PyModule_Create(&cclassy_module);
        if(!mod) return NULL;
        import_array();
        return mod;
    
}