#include "kmeans.hh"

#include <cmath>

KMeansGPU::KMeansGPU(int nb_clusters,
                     int nb_samples,
                     int nb_features,
                     int nb_iter,
                     std::string type) {
    this->nb_clusters = nb_clusters;
    this->nb_features = nb_features;
    this->nb_samples = nb_samples;
    this->nb_iter = nb_iter;
    this->type = type;

    // allocation of data
    cudaMallocManaged(&clusters, sizeof(int) * nb_clusters * nb_features);
    cudaCheckError();

    cudaMallocManaged(&data_clusters, sizeof(int) * nb_samples);
    cudaCheckError();
}

KMeansGPU::~KMeansGPU() {
    cudaFree(this->clusters);
    cudaFree(this->data_clusters);
}

// data: stape=(nb_samples, nb_features)
void KMeansGPU::fit(int* data) {
    // initialisation step
    dim3 blocks_();
    dim3 threads_();

    for (int i = 0; i < this->nb_iter; ++i) {
        // assotiate each sample to it's closest cluster
        find_closest_cluster_gpu<<<blocks_, threads_>>>(data, data_clusters,
                                                        clusters, nb_clusters,
                                                        nb_samples, nb_features);
        // compute cluster mean and set new clusters
        compute_cluster_mean();
    }
}

void KMeansGPU::compute_cluster_mean() {
    // TODO
}

// Compute Euclidian distance between proper vector of data and proper cluster
// dist: result of the function
// data: shape=(nb_samples, nb_features)
// clusters: shape=(nb_clusters, nb_features)
__device__ void dist(int* dist, int* data, int* clusters, int p1, int p2, int nb_features) {
    for (int i = 0; i < nb_features; ++i) {
        int pos_data = p1 * nb_features + i;
        int pos_cluster = p2 * nb_features + i;
        *dist += pow((data[pos_data] - custers[pos_cluster]), 2);
    }
    *dist = sqrt(*dist);
}

__global__ void find_closest_cluster_gpu(int* data,
                                         int* data_clusters,
                                         int* clusters,
                                         int nb_clusters,
                                         int nb_samples,
                                         int nb_features) {
    // TODO
}