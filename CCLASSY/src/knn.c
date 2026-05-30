#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    double distance;
    int label;
} Neighbor;

static int compare_neighbors(const void* a, const void* b){
    const Neighbor* na = (const Neighbor*)a;
    const Neighbor* nb = (const Neighbor*)b;
    if (na->distance < nb->distance) return -1;
    if (na->distance > nb->distance) return 1;
    return 0;
}

int* knn_predict(const double* X_train, const int* y_train,
    size_t n_train, size_t n_features,
    const double* X_test, size_t n_test, int k){

    int* predictions = (int*)malloc(n_test * sizeof(int));
    if (!predictions) return NULL;

    #pragma omp parallel
    {
        Neighbor* neighbors = (Neighbor*)malloc(n_train * sizeof(Neighbor));
        
        if (neighbors){
            #pragma omp for schedule(dynamic, 16)
            for (size_t i=0; i<n_test; i++){
                const double* x_query = X_test + i * n_features;
                for (size_t j=0; j<n_train; j++){
                    const double* x_train_j = X_train + j*n_features;
                    neighbors[j].distance = euclidean_distance(x_query, x_train_j, n_features);
                    neighbors[j].label = y_train[j];
                }
                qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

                int votes = 0;
                for (int m=0; m<k; m++){
                    votes += neighbors[m].label;
                }
                predictions[i] = (votes * 2 >= k) ? 1 : 0;
            }
            free(neighbors);
        }
    }
    return predictions;
}