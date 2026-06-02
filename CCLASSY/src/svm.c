#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    double* weights;
    double bias;
    size_t n_features;
} SVMModel;

SVMModel* svm_fit(const double* X_train, const int* y_train,
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate, double lambda)
{
    SVMModel* model = (SVMModel*)malloc(sizeof(SVMModel));
    if (!model) return NULL;

    model->n_features = n_features;
    model->bias = 0.0;
    model->weights = (double*)calloc(n_features, sizeof(double));
    if (!model->weights){
        free(model);
        return NULL;
    }

    double* dw = (double*)malloc(n_features * sizeof(double));
    if (!dw) {
        free(model->weights);
        free(model);
        return NULL;
    }

    for (int iter=0; iter<n_iterations; iter++){
        for(size_t j=0; j<n_features; j++){
            dw[j] = lambda * model->weights[j];
        }
        double db = 0.0;
        #pragma omp parallel for reduction(+:db) schedule(static)
        for(size_t i=0; i<n_train; i++){
            int y_svm = (y_train[i] == 1) ? 1 : -1;
            const double* xi = X_train + i*n_features;

            double z = model->bias;
            for (size_t j=0; j<n_features; j++){
                z += model->weights[j] * xi[j];
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
            model->weights[j] -= scale * dw[j];
        }
        model->bias -= scale*db;
    }
    free(dw);
    return model;
}

int* svm_predict(const SVMModel* model, const double* X_test, size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if (!predictions) return NULL;

    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<n_test; i++){
        double z = model->bias;
        const double* xi = X_test + i*model->n_features;
        for (size_t j=0; j<model->n_features; j++){
            z += model->weights[j] * xi[j];
        }
        predictions[i] = (z >= 0.0) ? 1 : 0;
    }
    return predictions;
}

void svm_free(SVMModel* model){
    if (!model) return;
    free(model->weights);
    free(model);
}