#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "common.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

int* knn_predict(
    const double* X_train, const int* y_train,
    size_t n_train, size_t n_features, 
    const double* X_test, size_t n_test, int k);

typedef struct{
    double* mean;
    double* var;
    double prior[2];
    size_t n_features;
} NBModel;

NBModel* nb_fit(const double* X_train, const int* y_train, size_t n_train, size_t n_features);
int* nb_predict(const NBModel* model, const double* X_test, size_t n_test);
void nb_free(NBModel* model);

typedef struct {
    double* weights;
    double bias;
    size_t n_features;
} LRModel;

LRModel* lr_fit(const double* X_train, const int* y_train, 
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate);
int* lr_predict(const LRModel* model, const double* X_test, size_t n_test);
void lr_free(LRModel* model);


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
    NBModel* model = nb_fit(X,y,n_samples, n_features);
    if(!model)
        return PyErr_Format(PyExc_RuntimeError,
        "nb_fit failed (memory allocation error)");
    PyObject* capsule = PyCapsule_New(model, "NBModel", NULL);
    if(!capsule){
        nb_free(model);
        return PyErr_Format(PyExc_RuntimeError,
        "failed to create capsule for model");
    }
    return capsule;
}

static PyObject* py_nb_predict(PyObject* self, PyObject* args){
    PyObject* capsule;
    PyArrayObject* X_test_arr;
    if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyArray_Type, &X_test_arr))
        return NULL;
    NBModel* model = (NBModel*)PyCapsule_GetPointer(capsule, "NBModel");
    if(!model)
        return PyErr_Format(PyExc_RuntimeError, "invalid model capsule");
    if (PyArray_TYPE(X_test_arr) != NPY_DOUBLE)
        return PyErr_Format(PyExc_TypeError, "X must be float");
    if (PyArray_NDIM(X_test_arr) != 2)
        return PyErr_Format(PyExc_ValueError, "X must be 2D");
    if (!PyArray_IS_C_CONTIGUOUS(X_test_arr))
        return PyErr_Format(PyExc_ValueError, "X must be C-contiguous");
    if ((size_t)PyArray_DIM(X_test_arr, 1) != model->n_features)
        return PyErr_Format(PyExc_ValueError, "X has wrong number of features");
    
    size_t n_test = (size_t)PyArray_DIM(X_test_arr, 0);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);
    int* predictions = nb_predict(model, X_test, n_test);
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

static PyObject* py_nb_free(PyObject* self, PyObject* args){
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return NULL;
    NBModel* model = (NBModel*)PyCapsule_GetPointer(capsule, "NBModel");
    if (model) nb_free(model);
    Py_RETURN_NONE;
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
    LRModel* model = lr_fit(X, y, n_samples, n_features,
        n_iterations, learning_rate);
    if (!model)
        return PyErr_Format(PyExc_RuntimeError, 
            "lr_fit_failed (memory allocation error)");
    PyObject* capsule = PyCapsule_New(model, "LRModel", NULL);
    if(!capsule) {
        lr_free(model);
        return PyErr_Format(PyExc_RuntimeError, 
            "failed to create capsule for model");
    }
    return capsule;
}


static PyObject* py_lr_predict(PyObject* self, PyObject* args){
    PyObject* capsule;
    PyArrayObject* X_test_arr;
    if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyArray_Type, &X_test_arr))
        return NULL;
    LRModel* model = (LRModel*)PyCapsule_GetPointer(capsule, "LRModel");
    if(!model)
        return PyErr_Format(PyExc_RuntimeError, "invalid model capsule");
    if (PyArray_TYPE(X_test_arr) != NPY_DOUBLE)
        return PyErr_Format(PyExc_TypeError, "X must be float");
    if (PyArray_NDIM(X_test_arr) != 2)
        return PyErr_Format(PyExc_ValueError, "X must be 2D");
    if (!PyArray_IS_C_CONTIGUOUS(X_test_arr))
        return PyErr_Format(PyExc_ValueError, "X must be C-contiguous");
    if ((size_t)PyArray_DIM(X_test_arr, 1) != model->n_features)
        return PyErr_Format(PyExc_ValueError, "X has wrong number of features");
    
    size_t n_test = (size_t)PyArray_DIM(X_test_arr, 0);
    const double* X_test = (const double*)PyArray_DATA(X_test_arr);
    int* predictions = lr_predict(model, X_test, n_test);
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

static PyObject* py_lr_free(PyObject* self, PyObject* args){
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return NULL;
    LRModel* model = (LRModel*)PyCapsule_GetPointer(capsule, "LRModel");
    if (model) lr_free(model);
    Py_RETURN_NONE;
}


static PyMethodDef cclassy_methods[] = {
    {"py_knn_predict", py_knn_predict, METH_VARARGS,"KNN prediction"},
    {"py_nb_fit", py_nb_fit, METH_VARARGS, "Naive Bayes fit"},
    {"py_nb_predict", py_nb_predict, METH_VARARGS, "Naive Bayes predict"},
    {"py_nb_free", py_nb_free, METH_VARARGS, "Naive Bayes free model"},
    {"py_lr_fit", py_lr_fit, METH_VARARGS, "Logistic Regression fit"},
    {"py_lr_predict", py_lr_predict, METH_VARARGS, "Logistic Regression predict"},
    {"py_lr_free", py_lr_free, METH_VARARGS, "Logistic Regression free model"},
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