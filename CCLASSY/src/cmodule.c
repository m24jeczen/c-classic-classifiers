#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "common.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

int* knn_predict(
    const double* X_train, const int* y_train,
    size_t n_train, size_t n_features, 
    const double* X_test, size_t n_test, int k);

int nb_fit(const double* X_train, const int* y_train, size_t n_train, size_t n_features,
            double* out_mean, double* out_var, double* out_prior);
int* nb_predict(const double* mean, const double* var, const double* prior,
                 size_t n_features, const double* X_test, size_t n_test);

double* lr_fit(const double* X_train, const int* y_train, 
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate,
    double* out_bias);
int* lr_predict(const double* weights, double bias, size_t n_features, const double* X_test, size_t n_test);


double* svm_fit(const double* X_train, const int* y_train, size_t n_train, size_t n_features,
    int n_iterations, double learning_rate, double lambda, double* out_bias);
int* svm_predict(const double* weights, double bias, size_t n_features, const double* X_test, size_t n_test);


static int parse_X_y(PyObject* args, 
    PyArrayObject** X_arr, PyArrayObject** y_arr,
    size_t* n_samples, size_t* n_features)
{
    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, X_arr, &PyArray_Type, y_arr))
        return 0;
    if (PyArray_TYPE(*X_arr) != NPY_DOUBLE || PyArray_TYPE(*y_arr) != NPY_INT )
    {
        PyErr_SetString(PyExc_TypeError, "X must be float64, y must be int32");
        return 0;
    }
    if(!PyArray_IS_C_CONTIGUOUS(*X_arr) || !PyArray_IS_C_CONTIGUOUS(*y_arr))
    {
        PyErr_SetString(PyExc_ValueError, "arrays must be C-contiguous");
        return 0;
    }
    *n_samples = (size_t)PyArray_DIM(*X_arr, 0);
    *n_features = (size_t)PyArray_DIM(*X_arr, 1);
    if ((size_t)PyArray_DIM(*y_arr, 0) != *n_samples)
    {
        PyErr_SetString(PyExc_ValueError, "X and y must have the same number of samples");
        return 0;
    }
    return 1;
}


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

static PyObject* py_nb_fit(PyObject* self, PyObject* args){
    PyArrayObject *X_arr, *y_arr;
    size_t n_samples, n_features;
    if (!parse_X_y(args, &X_arr, &y_arr, &n_samples, &n_features))
        return NULL;
    const double* X = (const double*)PyArray_DATA(X_arr);
    const int* y = (const int*) PyArray_DATA(y_arr);

    npy_intp dims_2d[2] = {2, (npy_intp)n_features};
    npy_intp dims_1d[1] = {2};

    PyObject* mean_arr = PyArray_ZEROS(2, dims_2d, NPY_DOUBLE, 0);
    PyObject* var_arr = PyArray_ZEROS(2, dims_2d, NPY_DOUBLE, 0);
    PyObject* prior_arr = PyArray_ZEROS(1, dims_1d, NPY_DOUBLE, 0);

    if(!mean_arr || !var_arr || !prior_arr){
        Py_XDECREF(mean_arr);
        Py_XDECREF(var_arr);
        Py_XDECREF(prior_arr);
        return PyErr_Format(PyExc_RuntimeError,
            "failed to allocate output arrays");
    }

    double* mean = (double*)PyArray_DATA((PyArrayObject*)mean_arr);
    double* var = (double*)PyArray_DATA((PyArrayObject*)var_arr);
    double* prior = (double*)PyArray_DATA((PyArrayObject*)prior_arr);

    if (!nb_fit(X,y, n_samples, n_features, mean, var, prior)){
        Py_DECREF(mean_arr);
        Py_DECREF(var_arr);
        Py_DECREF(prior_arr);
        return PyErr_Format(PyExc_RuntimeError, "nb_fit failed");
    }

    PyObject* result = PyTuple_Pack(3,mean_arr, var_arr, prior_arr);
    Py_DECREF(mean_arr);
    Py_DECREF(var_arr);
    Py_DECREF(prior_arr);
    return result;

}

static PyObject* py_nb_predict(PyObject* self, PyObject* args){
    PyArrayObject *mean_arr, *var_arr, *prior_arr, *X_test_arr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
        &PyArray_Type, &mean_arr,
        &PyArray_Type, &var_arr,
        &PyArray_Type, &prior_arr,
        &PyArray_Type, &X_test_arr))
        return NULL;

    if (PyArray_TYPE(mean_arr) != NPY_DOUBLE ||
        PyArray_TYPE(var_arr) != NPY_DOUBLE ||
        PyArray_TYPE(prior_arr) != NPY_DOUBLE ||
        PyArray_TYPE(X_test_arr) != NPY_DOUBLE)
        return PyErr_Format(PyExc_TypeError, "all arrays must be float64");

    
    if (!PyArray_IS_C_CONTIGUOUS(mean_arr) ||
        !PyArray_IS_C_CONTIGUOUS(var_arr) ||
        !PyArray_IS_C_CONTIGUOUS(prior_arr) ||
        !PyArray_IS_C_CONTIGUOUS(X_test_arr))
        return PyErr_Format(PyExc_TypeError, "all arrays must be C-contiguous");

    if (PyArray_NDIM(mean_arr) != 2 ||
        PyArray_NDIM(var_arr) != 2 ||
        PyArray_NDIM(prior_arr) != 1 ||
        PyArray_NDIM(X_test_arr) != 2)
        return PyErr_Format(PyExc_TypeError, 
            "mean,var, X must be 2D, prior must be 1D");

    size_t n_features = (size_t)PyArray_DIM(mean_arr,1);
    size_t n_test = (size_t)PyArray_DIM(X_test_arr,0);

    if ((size_t)PyArray_DIM(X_test_arr, 1) != n_features)
            return PyErr_Format(PyExc_ValueError, 
            "X has wrong number of features"); 

    const double* mean = (double*)PyArray_DATA((PyArrayObject*)mean_arr);
    const double* var = (double*)PyArray_DATA((PyArrayObject*)var_arr);
    const double* prior = (double*)PyArray_DATA((PyArrayObject*)prior_arr);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);

    int* predictions = nb_predict(mean, var, prior, n_features, X_test, n_test);
    if(!predictions)
        return PyErr_Format(PyExc_RuntimeError, 
            "nb_predict failed (memory allocation error)");

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

static PyObject* py_lr_fit(PyObject* self, PyObject* args){
    PyArrayObject *X_arr, *y_arr;
    size_t n_samples, n_features;
    int n_iterations;
    double learning_rate;
    if(!PyArg_ParseTuple(args, "O!O!id",
        &PyArray_Type, &X_arr,
        &PyArray_Type, &y_arr,
        &n_iterations,
        &learning_rate))
    return NULL;

    if (PyArray_TYPE(X_arr) != NPY_DOUBLE ||
        PyArray_TYPE(y_arr) != NPY_INT)
        return PyErr_Format(PyExc_TypeError,
            "X must be float64, y must be int32");

    if (!PyArray_IS_C_CONTIGUOUS(X_arr) ||
        !PyArray_IS_C_CONTIGUOUS(y_arr))
        return PyErr_Format(PyExc_ValueError,
        "arrays must be C-contiguous");

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(y_arr) != 1)
            return PyErr_Format(PyExc_ValueError, 
            "X must be 2D, y must be 1D");

    n_samples = (size_t)PyArray_DIM(X_arr, 0);
    n_features = (size_t)PyArray_DIM(X_arr, 1);
    if ((size_t)PyArray_DIM(y_arr, 0) != n_samples)
        return PyErr_Format(PyExc_ValueError, "X and y must have the same number of samples");

    if(n_iterations <= 0)
        return PyErr_Format(PyExc_ValueError, 
            "n_iterations must be > 0");
    if(learning_rate <= 0.0)
        return PyErr_Format(PyExc_ValueError, 
            "learning_rate must be >0");
    const double* X = (const double*)PyArray_DATA(X_arr);
    const int* y = (const int*)PyArray_DATA(y_arr);

    double out_bias = 0.0;
    double* weights = lr_fit(X,y, n_samples, n_features, n_iterations, learning_rate, &out_bias);

    if(!weights) 
        return PyErr_Format(PyExc_RuntimeError, 
            "lr_fit failed (memory allocation error)");

    npy_intp dims[1] = {(npy_intp)n_features};
    PyObject* weights_arr = PyArray_SimpleNewFromData(
        1, dims, NPY_DOUBLE, weights);
    if (!weights_arr) {
        free(weights);
        return PyErr_Format(PyExc_RuntimeError, 
            "failed to create weights array");
    }

    PyArray_ENABLEFLAGS((PyArrayObject*)weights_arr, NPY_ARRAY_OWNDATA);
    PyObject* result = PyTuple_Pack(2,
        weights_arr,
        PyFloat_FromDouble(out_bias));
    Py_DECREF(weights_arr);
    return result;

}


static PyObject* py_lr_predict(PyObject* self, PyObject* args){
    PyArrayObject *weights_arr, *X_test_arr;
    double bias;

    if (!PyArg_ParseTuple(args, "O!dO!", 
        &PyArray_Type, &weights_arr, &bias, &PyArray_Type, &X_test_arr))
        return NULL;

    if (PyArray_TYPE(X_test_arr) != NPY_DOUBLE ||
        PyArray_TYPE(weights_arr) != NPY_DOUBLE)
        return PyErr_Format(PyExc_TypeError, "weights and X must be float64");
    if (PyArray_NDIM(X_test_arr) != 2 || PyArray_NDIM(weights_arr) != 1)
        return PyErr_Format(PyExc_ValueError, "weights must be 1D, X must be 2D");
    if (!PyArray_IS_C_CONTIGUOUS(X_test_arr) ||
        !PyArray_IS_C_CONTIGUOUS(weights_arr))
        return PyErr_Format(PyExc_ValueError, "X and weights must be C-contiguous");

    size_t n_features = (size_t)PyArray_DIM(weights_arr, 0);
    size_t n_test = (size_t)PyArray_DIM(X_test_arr, 0);

    if ((size_t)PyArray_DIM(X_test_arr, 1) != n_features)
        return PyErr_Format(PyExc_ValueError, "X has wrong number of features");
    
    const double* weights = (const double*)PyArray_DATA(weights_arr);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);
    int* predictions = lr_predict(weights, bias, n_features, X_test, n_test);
    if(!predictions)
        return PyErr_Format(PyExc_RuntimeError, 
            "lr_predict failed (memory allocation error)");

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


static PyObject* py_svm_fit(PyObject* self, PyObject* args){
    PyArrayObject *X_arr, *y_arr;
    size_t n_samples, n_features;
    int n_iterations;
    double learning_rate, lambda;
    if(!PyArg_ParseTuple(args, "O!O!idd",
        &PyArray_Type, &X_arr,
        &PyArray_Type, &y_arr,
        &n_iterations,
        &learning_rate,
        &lambda))
    return NULL;

    if (PyArray_TYPE(X_arr) != NPY_DOUBLE ||
        PyArray_TYPE(y_arr) != NPY_INT)
        return PyErr_Format(PyExc_TypeError,
            "X must be float64, y must be int32");

    if (!PyArray_IS_C_CONTIGUOUS(X_arr) ||
        !PyArray_IS_C_CONTIGUOUS(y_arr))
        return PyErr_Format(PyExc_ValueError,
        "arrays must be C-contiguous");

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(y_arr) != 1)
            return PyErr_Format(PyExc_ValueError, 
            "X must be 2D, y must be 1D");

    n_samples = (size_t)PyArray_DIM(X_arr, 0);
    n_features = (size_t)PyArray_DIM(X_arr, 1);
    if ((size_t)PyArray_DIM(y_arr, 0) != n_samples)
        return PyErr_Format(PyExc_ValueError, "X and y must have the same number of samples");

    if(n_iterations <= 0)
        return PyErr_Format(PyExc_ValueError, 
            "n_iterations must be > 0");
    if(learning_rate <= 0.0)
        return PyErr_Format(PyExc_ValueError, 
            "learning_rate must be >0");
    if(lambda <= 0.0)
        return PyErr_Format(PyExc_ValueError, 
            "lambda must be >0");

    const double* X = (const double*)PyArray_DATA(X_arr);
    const int* y = (const int*)PyArray_DATA(y_arr);

    double out_bias = 0.0;
    double* weights = svm_fit(X,y, n_samples, n_features, n_iterations, learning_rate, lambda, &out_bias);

    if(!weights) 
        return PyErr_Format(PyExc_RuntimeError, 
            "svm_fit failed (memory allocation error)");

    npy_intp dims[1] = {(npy_intp)n_features};
    PyObject* weights_arr = PyArray_SimpleNewFromData(
        1, dims, NPY_DOUBLE, weights);
    if (!weights_arr) {
        free(weights);
        return PyErr_Format(PyExc_RuntimeError, 
            "failed to create weights array");
    }

    PyArray_ENABLEFLAGS((PyArrayObject*)weights_arr, NPY_ARRAY_OWNDATA);
    PyObject* result = PyTuple_Pack(2,
        weights_arr,
        PyFloat_FromDouble(out_bias));
    Py_DECREF(weights_arr);
    return result;

}

static PyObject* py_svm_predict(PyObject* self, PyObject* args){
    PyArrayObject *weights_arr, *X_test_arr;
    double bias;

    if (!PyArg_ParseTuple(args, "O!dO!", 
        &PyArray_Type, &weights_arr, &bias, &PyArray_Type, &X_test_arr))
        return NULL;

    if (PyArray_TYPE(X_test_arr) != NPY_DOUBLE ||
        PyArray_TYPE(weights_arr) != NPY_DOUBLE)
        return PyErr_Format(PyExc_TypeError, "weights and X must be float64");
    if (PyArray_NDIM(X_test_arr) != 2 || PyArray_NDIM(weights_arr) != 1)
        return PyErr_Format(PyExc_ValueError, "weights must be 1D, X must be 2D");
    if (!PyArray_IS_C_CONTIGUOUS(X_test_arr) ||
        !PyArray_IS_C_CONTIGUOUS(weights_arr))
        return PyErr_Format(PyExc_ValueError, "X and weights must be C-contiguous");

    size_t n_features = (size_t)PyArray_DIM(weights_arr, 0);
    size_t n_test = (size_t)PyArray_DIM(X_test_arr, 0);

    if ((size_t)PyArray_DIM(X_test_arr, 1) != n_features)
        return PyErr_Format(PyExc_ValueError, "X has wrong number of features");
    
    const double* weights = (const double*)PyArray_DATA(weights_arr);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);
    int* predictions = svm_predict(weights, bias, n_features, X_test, n_test);
    if(!predictions)
        return PyErr_Format(PyExc_RuntimeError, 
            "svm_predict failed (memory allocation error)");

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
    {"py_knn_predict", py_knn_predict, METH_VARARGS,"KNN prediction"},
    {"py_nb_fit", py_nb_fit, METH_VARARGS, "Naive Bayes fit"},
    {"py_nb_predict", py_nb_predict, METH_VARARGS, "Naive Bayes predict"},
    {"py_lr_fit", py_lr_fit, METH_VARARGS, "Logistic Regression fit"},
    {"py_lr_predict", py_lr_predict, METH_VARARGS, "Logistic Regression predict"},
    {"py_svm_fit", py_svm_fit, METH_VARARGS, "SVM fit"},
    {"py_svm_predict", py_svm_predict, METH_VARARGS, "SVM predict"},
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