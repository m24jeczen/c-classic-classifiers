#ifndef CCLASSY_COMMON_H
#define CCLASSY_COMMON_H

#include <stdlib.h>
#include <math.h>

typedef struct {
    double* X;
    int* y;
    size_t n_samples;
    size_t n_features;
} Dataset;

double euclidean_distance(const double* a, const double* b, size_t n_features);

#endif