#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    double* weights;
    double bias;
    size_t n_features;
} LRModel;

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

LRModel* lr_fit(
    const double* X_train, const int* y_train,
    size_t n_train, size_t n_features,
    int n_iterations, double learning_rate)
{
    LRModel* model = (LRModel*)malloc(sizeof(LRModel));
    if (!model) return NULL;

    model->n_features = n_features;
    model->bias = 0.0;
    model->weights = (double*)calloc(n_features, sizeof(double));
    if (!model->weights){
        free(model);
        return NULL;
    }

    double* errors = (double*)malloc(n_train * sizeof(double));
    if (!errors) {
        free(model->weights);
        free(model);
        return NULL;
    }

    double* dw = (double*)malloc(n_features * sizeof(double));
    if (!dw){
        free(errors);
        free(model->weights);
        free(model);
        return NULL;
    }

    for (int iter=0; iter<n_iterations; iter++){

        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<n_train; i++){
            double z = model->bias;
            const double* xi = X_train + i*n_features;
            for (size_t j=0; j<n_features; j++){
                z += model->weights[j] * xi[j];
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
            model->weights[j] -= scale * dw[j];
        }
        model -> bias -= scale*db;
    }
    free(errors);
    free(dw);
    return model;
}

int* lr_predict(const LRModel* model, const double* X_test,
    size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if(!predictions) return NULL;

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<n_test; i++){
        double z = model->bias;
        const double* xi = X_test + i*model->n_features;
        for(size_t j=0; j < model->n_features; j++){
            z += model->weights[j] * xi[j];
        }
        predictions[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
    }
    return predictions;
}

void lr_free(LRModel* model){
    if (!model) return;
    free(model->weights);
    free(model);
}