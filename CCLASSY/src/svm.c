#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif


double* svm_fit(const double* X_train, const int* y_train,
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate, double lambda,
    double* out_bias)
{
    double* weights = (double*)calloc(n_features, sizeof(double));
    if (!weights) return NULL;

    double bias = 0.0;

    double* dw = (double*)malloc(n_features * sizeof(double));
    if (!dw) {
        free(weights);
        return NULL;
    }

    for (int iter=0; iter<n_iterations; iter++){
        for(size_t j=0; j<n_features; j++){
            dw[j] = lambda * weights[j];
        }
        double db = 0.0;
        #pragma omp parallel for reduction(+:db) schedule(static)
        for(size_t i=0; i<n_train; i++){
            int y_svm = (y_train[i] == 1) ? 1 : -1;
            const double* xi = X_train + i*n_features;
            double z = bias;

            for (size_t j=0; j<n_features; j++){
                z += weights[j] * xi[j];
            }

            if (y_svm * z < 1.0){
                #pragma omp simd
                for (size_t j=0; j<n_features; j++) {
                    #pragma omp atomic
                    dw[j] -= y_svm * xi[j];
                }
                db -= y_svm;
            }
        }

        double scale = learning_rate/(double)n_train;
        for (size_t j=0; j<n_features; j++){
            weights[j] -= scale * dw[j];
        }
        bias -= scale*db;
    }
    *out_bias = bias;
    free(dw);
    return weights;
}

int* svm_predict(const double* weights, double bias,
        size_t n_features, const double* X_test, size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if (!predictions) return NULL;

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<n_test; i++){
        double z = bias;
        const double* xi = X_test + i*n_features;
        for (size_t j=0; j<n_features; j++){
            z += weights[j] * xi[j];
        }
        predictions[i] = (z >= 0.0) ? 1 : 0;
    }
    return predictions;
}
