#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    double* mean;
    double* var;
    double prior[2];
    size_t n_features;
} NBModel;

static double log_gaussian(double x, double mean, double var) {
    return -0.5 * log(2.0*M_PI * var) - ((x-mean) * (x-mean)) / (2.0 * var);
}

NBModel* nb_fit(const double* X_train, const int* y_train, 
    size_t n_train, size_t n_features)
{
    NBModel* model = (NBModel*)malloc(sizeof(NBModel));
    if(!model) return NULL;

    model->n_features = n_features;
    model->mean = (double*)calloc(2*n_features, sizeof(double));
    model->var = (double*)calloc(2*n_features, sizeof(double));
    if (!model->mean || !model->var){
        free(model->mean);
        free(model->var);
        free(model);
        return NULL;
    }
    size_t count[2] = {0,0};
    for(size_t i=0; i<n_train; i++){
        int c=y_train[i];
        count[c]++;
        for(size_t j=0; j<n_features; j++){
            model->mean[c*n_features + j] += X_train[i*n_features+j];
        }
    }

    for(int c=0; c<2; c++){
        model->prior[c] = (double)count[c] / (double)n_train;
        if(count[c] == 0) continue;
        for(size_t j=0; j<n_features; j++){
            model->mean[c*n_features + j] /= (double)count[c];
        }
    }

    for(size_t i=0; i<n_train; i++){
        int c=y_train[i];
        for (size_t j=0; j<n_features; j++){
            double diff = X_train[i*n_features + j] - model->mean[c*n_features + j];
            model->var[c*n_features + j] += diff*diff;
        }
    }
    for(int c=0; c<2; c++){
        if (count[c]==0) continue;
        for(size_t j=0; j<n_features; j++){
            model->var[c*n_features + j] /= (double)count[c];
            if (model->var[c*n_features + j] < 1e-9)
                model->var[c*n_features + j] = 1e-9;
        }
    }
    return model;
}

int* nb_predict(
    const NBModel* model,
    const double* X_test, size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if (!predictions) return NULL;

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i=0; i<n_test; i++){
        const double* x = X_test + i * model->n_features;
        double log_scores[2];

        for(int c=0; c<2; c++){
            log_scores[c] = log(model->prior[c]);
            for(size_t j=0; j < model->n_features; j++){
                log_scores[c] += log_gaussian(
                    x[j],
                    model->mean[c * model->n_features + j],
                    model->var[c * model->n_features + j]);
            }
        }
        predictions[i] = (log_scores[1] > log_scores[0]) ? 1 : 0;
    }
    return predictions;
}

void nb_free(NBModel* model){
    if (!model) return;
    free(model->mean);
    free(model->var);
    free(model);
}