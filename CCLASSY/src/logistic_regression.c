#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double* lr_fit(
    const double* X_train, const int* y_train,
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate,
    double* out_bias)
{
    double* weights = (double*)malloc(sizeof(double));
    if (!weights) return NULL;

    double bias = 0.0;

    double* errors = (double*)malloc(n_train * sizeof(double));
    if (!errors) {
        free(weights);
        return NULL;
    }

    double* dw = (double*)malloc(n_features * sizeof(double));
    if (!dw){
        free(errors);
        free(weights);
        return NULL;
    }

    for (int iter=0; iter<n_iterations; iter++){

        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<n_train; i++){
            double z = bias;
            const double* xi = X_train + i*n_features;
            for (size_t j=0; j<n_features; j++){
                z += weights[j] * xi[j];
            }
            errors[i] = sigmoid(z) - (double)y_train[i];
        }
        for(size_t j=0; j<n_features; j++){
            dw[j] = 0.0;
        }
        double db = 0.0;

        #pragma omp parallel for reduction(+:db) schedule(static)
        for(size_t i=0; i<n_train; i++){
            const double* xi = X_train + i*n_features;
            #pragma omp simd
            for(size_t j=0; j<n_features; j++){
                dw[j] += errors[i] * xi[j];
            }
            db += errors[i];
        }

        double scale = learning_rate / (double)n_train;
        for(size_t j=0; j<n_features; j++){
            weights[j] -= scale * dw[j];
        }
        bias -= scale*db;
    }
    *out_bias = bias;
    free(errors);
    free(dw);
    return weights;
}


int* lr_predict(const double* weights, double bias, size_t n_features, const double* X_test,
    size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if(!predictions) return NULL;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<n_test; i++){
        double z = bias;
        const double* xi = X_test + i*n_features;
        for(size_t j=0; j < n_features; j++){
            z += weights[j] * xi[j];
        }
        predictions[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
    }
    return predictions;
}
