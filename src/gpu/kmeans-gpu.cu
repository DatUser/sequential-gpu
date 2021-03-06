#include "kmeans-gpu.hh"

#include <cmath>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

__device__ void dist(float* dist, int* data,
                     float* clusters, int p1,
                     int p2, int nb_features);

__global__ void find_closest_cluster_gpu(float* data,
                                         int* data_clusters,
                                         float* clusters,
                                         int nb_clusters,
                                         int nb_samples,
                                         int nb_features);

__global__ void set_val(float* data, float val, int size);

__global__ void init_clusters_gpu(float* clusters, float* data,
                                    int nb_clusters, int nb_features,
                                    int sample_ID, int cluster_ID);

KMeansGPU::KMeansGPU(int nb_clusters,
                     int nb_samples,
                     int nb_features,
                     int nb_iter,
                     std::string type,
                     bool verbose) {
    this->nb_clusters = nb_clusters;
    this->nb_features = nb_features;
    this->nb_samples = nb_samples;
    this->nb_iter = nb_iter;
    this->type = type;
    this->verbose = verbose;

    // allocation of data
    cudaMallocManaged(&clusters, sizeof(float) * nb_clusters * nb_features);
    cudaCheckError();
    cudaMemset(clusters, 0, sizeof(float) * nb_clusters * nb_features);
    cudaCheckError();

    cudaMallocManaged(&data_clusters, sizeof(int) * nb_samples);
    cudaCheckError();
    cudaMemset(data_clusters, 0, sizeof(int) * nb_samples);
    cudaCheckError();
}

KMeansGPU::~KMeansGPU() {
    cudaFree(this->clusters);
    cudaCheckError();
    cudaFree(this->data_clusters);
    cudaCheckError();
}

// Init one cluster using the forgy initialization
void KMeansGPU::forgy_cluster_init(float* data, int cluster_ID, std::unordered_map<int, bool>& map) {
    // get random id
    int sample_id = rand() % nb_samples;
    while (map.find(sample_id) != map.end()) {
        //std::cout << sample_id << '\n';
        sample_id = rand() % nb_samples;
    }
    map[sample_id] = true;

    //for (int i = 0; i < nb_features; ++i) {
    //    this->clusters[cluster_ID * nb_features + i] = data[sample_id * nb_features + i];
    //}
    int nb_blocks = 30;
    dim3 blocks_(nb_blocks);
    dim3 threads_((nb_features + nb_features) / nb_blocks);
    init_clusters_gpu<<<blocks_, threads_>>>(clusters, data, nb_clusters, nb_features, sample_id, cluster_ID);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

__global__ void init_clusters_gpu(float* clusters, float* data,
                                    int nb_clusters, int nb_features,
                                    int sample_ID, int cluster_ID) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nb_clusters * nb_features)
        return;

    clusters[cluster_ID * nb_features + idx] = data[sample_ID * nb_features + idx];
}

// Init all the clusters using forgy initialization
void KMeansGPU::forgy_clusters_init(float* data) {
    std::unordered_map<int, bool> map;
    for (int i = 0; i < this->nb_clusters; ++i) {
        forgy_cluster_init(data, i, map);
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

    int nb_blocks = 200;
    dim3 blocks_(nb_blocks);
    dim3 threads_((nb_samples + nb_blocks) / nb_blocks);

    for (int i = 0; i < this->nb_iter; ++i) {
        if (verbose)
            std::cout << "It nb. " << i << '\n';

        // assotiate each sample to it's closest cluster
        find_closest_cluster_gpu<<<blocks_, threads_>>>(data, data_clusters,
                                            clusters, nb_clusters,
                                            nb_samples, nb_features);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();

        // compute cluster mean and set new clusters
        compute_clusters_mean(data);
    }
}

void KMeansGPU::display_clusters() {
    std::cout << "\nClusters:\n";
    // display centroid init
    for (int i = 0; i < nb_clusters; ++i) {
        for (int j = 0; j < nb_features; ++j) {
            std::cout << (float) clusters[i * nb_features + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "\n\n";
}

void KMeansGPU::to_csv(const char* filepath, const char* sep, int nb_cols) {
    std::ofstream csv_ofstream(filepath);

    for (int i = 0; i < nb_samples / nb_cols; ++i) {
        for (int j = 0; j < nb_cols - 1; ++j) {
            csv_ofstream << (float) data_clusters[i * nb_cols + j] << sep;
        }
        csv_ofstream << (float) data_clusters[i * nb_cols + nb_cols - 1] << '\n';
    }

    csv_ofstream.close();
}

// When it is necessary to update clusters
void KMeansGPU::compute_clusters_mean(float* data) {
    // allocate data for keeping info of number of samples per clusters
    int* samples_histo;
    samples_histo = (int*) calloc(nb_clusters, sizeof(int));

    // set cluster to 0
    //for (int i = 0; i < nb_clusters * nb_features; ++i) {
    //    clusters[i] = 0.0;
    //}
    int s = nb_clusters * nb_features;
    int nb_blocks = 100;
    dim3 blocks_(nb_blocks);
    dim3 threads_((s + nb_blocks) / nb_blocks);
    set_val<<<blocks_, threads_>>>(clusters, 0.0, nb_clusters * nb_features);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

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
        if (samples_histo[k] == 0)
            continue;
        for (int l = 0; l < nb_features; ++l) {
            clusters[k * nb_features + l] /= samples_histo[k];
        }
    }

    free(samples_histo);
}

__global__ void set_val(float* data, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    data[idx] = val;
}

float KMeansGPU::dist(float* data, int p1, int p2) {
    float dist = 0.0;
    for (int i = 0; i < nb_features; ++i) {
        int pos_data = p1 * nb_features + i;
        int pos_cluster = p2 * nb_features + i;
        dist += pow((data[pos_data] - clusters[pos_cluster]), 2);
    }

    return sqrt(dist);
}

void KMeansGPU::find_closest_cluster(float* data, int sample_ID) {
    int curr_min_dist = dist(data, sample_ID, 0);
    int curr_min_cluster = 0;
    for (int i = 1; i < nb_clusters; ++i) {
        float cluster_dist;
        cluster_dist = dist(data, sample_ID, i);

        if (cluster_dist < curr_min_dist) {
            curr_min_cluster = i;
            curr_min_dist = cluster_dist;
        }
    }

    // set the min cluster idx
    data_clusters[sample_ID] = curr_min_cluster;
}

// Compute Euclidian distance between proper vector of data and proper cluster
// dist: result of the function
// data: shape=(nb_samples, nb_features)
// clusters: shape=(nb_clusters, nb_features)
// p1: position (row-wise) in 'data'
// p2: position (row-wise) in 'clusters'
// nb_features: number of features of the data (number of dimensions)
__device__ float dist(float* data,
                     float* clusters, int p1,
                     int p2, int nb_features) {
    float dist = 0.0;
    for (int i = 0; i < nb_features; ++i) {
        int pos_data = p1 * nb_features + i;
        int pos_cluster = p2 * nb_features + i;
        dist += pow((data[pos_data] - clusters[pos_cluster]), 2);
    }

    return sqrt(dist);
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

    int curr_min_dist = dist(data, clusters, idx, 0, nb_features);
    int curr_min_cluster = 0;
    for (int i = 1; i < nb_clusters; ++i) {
        float cluster_dist;
        cluster_dist = dist(data, clusters, idx, i, nb_features);

        if (cluster_dist < curr_min_dist) {
            curr_min_cluster = i;
            curr_min_dist = cluster_dist;
        }
    }

    // set the min cluster idx
    data_clusters[idx] = curr_min_cluster;
}