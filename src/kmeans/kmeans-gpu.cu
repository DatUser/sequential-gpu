#include "kmeans-gpu.hh"

#include <cmath>
#include <time.h>

__device__ void dist(float* dist, int* data,
                     float* clusters, int p1,
                     int p2, int nb_features);

__global__ void find_closest_cluster_gpu(float* data,
                                         int* data_clusters,
                                         float* clusters,
                                         int nb_clusters,
                                         int nb_samples,
                                         int nb_features);
 

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
    cudaMallocManaged(&clusters, sizeof(float) * nb_clusters * nb_features);
    cudaCheckError();

    cudaMallocManaged(&data_clusters, sizeof(int) * nb_samples);
    cudaCheckError();
}

KMeansGPU::~KMeansGPU() {
    cudaFree(this->clusters);
    cudaFree(this->data_clusters);
}

// Init one cluster using the forgy initialization
void KMeansGPU::forgy_cluster_init(float* data, int cluster_ID) {
    // get random id
    int sample_id = rand() % nb_samples;

    for (int i = 0; i < nb_features; ++i) {
        this->clusters[cluster_ID * nb_features + i] = data[sample_id * nb_features + i];
    }
}

// Init all the clusters using forgy initialization
void KMeansGPU::forgy_clusters_init(float* data) {
    for (int i = 0; i < this->nb_clusters; ++i) {
        forgy_cluster_init(data, i);
    }
}

// data: stape=(nb_samples, nb_features)
void KMeansGPU::fit(float* data) {
    // initialisation step
    // TODO change initialization
    // for now select random points
    srand(time(NULL));

    // forgy init
    forgy_clusters_init(data);

    int nb_blocks = 50;
    dim3 blocks_(nb_blocks);
    dim3 threads_((nb_samples + nb_blocks) / nb_blocks);

    for (int i = 0; i < this->nb_iter; ++i) {
        // assotiate each sample to it's closest cluster
        find_closest_cluster_gpu<<<blocks_, threads_>>>(data, data_clusters,
                                                        clusters, nb_clusters,
                                                        nb_samples, nb_features);
        // compute cluster mean and set new clusters
        compute_clusters_mean(data);
    }
}

// Compute the mean of all clusters
// When it is necessary to update clusters
void KMeansGPU::compute_clusters_mean(float* data) {
    // allocate data for keeping info of number of samples per clusters
    int* samples_histo;
    cudaMallocManaged(&samples_histo, sizeof(int) * nb_clusters);
    cudaCheckError();

    // set cluster to 0
    for (int i = 0; i < nb_clusters * nb_features; ++i) {
        clusters[i] = 0;
    }

    // compute the mean of each cluster
    for (int i = 0; i < nb_samples; ++i) {
        int cluster_ID = data_clusters[i];
        samples_histo[cluster_ID]++;
        for (int j = 0; j < nb_features; ++j) {
            clusters[cluster_ID * nb_features + j] += data[i * nb_features + j];
        }
    }

    // division of the mean
    for (int k = 0; k < nb_clusters; ++k) {
        for (int l = 0; l < nb_features; ++l) {
            clusters[k * nb_features + l] /= samples_histo[k];
        }
    }

    cudaFree(samples_histo);
    cudaCheckError();
}

// Compute Euclidian distance between proper vector of data and proper cluster
// dist: result of the function
// data: shape=(nb_samples, nb_features)
// clusters: shape=(nb_clusters, nb_features)
// p1: position (row-wise) in 'data'
// p2: position (row-wise) in 'clusters'
// nb_features: number of features of the data (number of dimensions)
__device__ void dist(float* dist, float* data,
                     float* clusters, int p1,
                     int p2, int nb_features) {
    for (int i = 0; i < nb_features; ++i) {
        int pos_data = p1 * nb_features + i;
        int pos_cluster = p2 * nb_features + i;
        *dist += pow((data[pos_data] - clusters[pos_cluster]), 2);
    }
    *dist = sqrt(*dist);
}

// For one sample find the clusest cluster on GPU
__global__ void find_closest_cluster_gpu(float* data,
                                         int* data_clusters,
                                         float* clusters,
                                         int nb_clusters,
                                         int nb_samples,
                                         int nb_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nb_samples)
        return;

    int curr_min_dist = 0;
    int curr_min_cluster = 0;
    for (int i = 0; i < nb_clusters; ++i) {
        float cluster_dist;
        dist(&cluster_dist, data, clusters, idx, i, nb_features);

        if (i == 0 || cluster_dist < curr_min_dist) {
            curr_min_cluster = i;
            curr_min_dist = cluster_dist;
        }
    }

    // set the min cluster idx
    data_clusters[idx] = curr_min_cluster;
}