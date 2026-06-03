#define _USE_MATH_DEFINES
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double log_gaussian(double x, double mean, double var) {
    return -0.5 * log(2.0*M_PI * var) - ((x-mean) * (x-mean)) / (2.0 * var);
}

int nb_fit(const double* X_train, const int* y_train, 
    size_t n_train, size_t n_features,
    double* out_mean, double* out_var, double* out_prior)
{

    size_t count[2] = {0,0};

    for(size_t j=0; j<2*n_features; j++){
        out_mean[j] = 0.0;
        out_var[j] = 0.0;
    }
    out_prior[0] = 0.0;
    out_prior[1] = 0.0;

    for(size_t i=0; i<n_train; i++){
        int c=y_train[i];
        count[c]++;
        for(size_t j=0; j<n_features; j++){
            out_mean[c*n_features + j] += X_train[i*n_features+j];
        }
    }

    for(int c=0; c<2; c++){
        out_prior[c] = (double)count[c] / (double)n_train;
        if(count[c] == 0) continue;
        for(size_t j=0; j<n_features; j++){
            out_mean[c*n_features + j] /= (double)count[c];
        }
    }

    for(size_t i=0; i<n_train; i++){
        int c=y_train[i];
        for (size_t j=0; j<n_features; j++){
            double diff = X_train[i*n_features + j] - out_mean[c*n_features + j];
            out_var[c*n_features + j] += diff*diff;
        }
    }
    for(int c=0; c<2; c++){
        if (count[c]==0) continue;
        for(size_t j=0; j<n_features; j++){
            out_var[c*n_features + j] /= (double)count[c];
            if (out_var[c*n_features + j] < 1e-9)
                out_var[c*n_features + j] = 1e-9;
        }
    }
    return 1;
}

int* nb_predict(
    const double* mean, const double* var, const double* prior,
    size_t n_features,
    const double* X_test, size_t n_test)
{
    int* predictions = (int*)malloc(n_test * sizeof(int));
    if (!predictions) return NULL;

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i=0; i<n_test; i++){
        const double* x = X_test + i * n_features;
        double log_scores[2];

        for(int c=0; c<2; c++){
            log_scores[c] = log(prior[c]);
            for(size_t j=0; j < n_features; j++){
                log_scores[c] += log_gaussian(
                    x[j],
                    mean[c * n_features + j],
                    var[c * n_features + j]);
            }
        }
        predictions[i] = (log_scores[1] > log_scores[0]) ? 1 : 0;
    }
    return predictions;
}
