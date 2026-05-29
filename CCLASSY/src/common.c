#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

double euclidean_distance(const double* a, const double* b, size_t n_features) {
    double sum=0.0;
    for (size_t i=0; i<n_features; i++){
        double diff = a[i] - b[i];
        sum += diff*diff;
    }
    return sqrt(sum);
}