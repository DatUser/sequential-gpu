#pragma once

#define cudaCheckError() {                                                                   \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

#include <string>

class KMeansGPU {
    public:
        KMeansGPU(int nb_clusters,
                  int nb_samples,
                  int nb_features,
                  int nb_iter,
                  std::string type);
        ~KMeansGPU();

        void fit(float* data);

        int nb_features;
        int nb_clusters;
        int nb_samples;
        int nb_iter;
        std::string type;

        // clusters data
        // shape=(nb_clusters, nb_features)
        float* clusters;

        // Each samples is associated to a cluster
        // shape=(nb_samples, )
        int* data_clusters;

    private:
        void forgy_cluster_init(float* data, int cluster_ID);
        void forgy_clusters_init(float* data);
        void compute_clusters_mean(float* data);
};