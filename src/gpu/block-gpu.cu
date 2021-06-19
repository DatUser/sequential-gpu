#include "block-gpu.hh"

__global__
void compute_texton_block_gpu(unsigned char* textons, unsigned char* blocks_img,
                              int block_size, int window_size, int nb_blocks) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;


    if (i < nb_blocks && j < block_size && k < block_size)
    {
      compute_pixel_texton_gpu(i, j, k, textons, blocks_img,
                               block_size, window_size, nb_blocks);
    }
}

// i <= nb_blocks
// j <= block_size
// k <= block_size
__device__
void compute_pixel_texton_gpu(int i, int j, int k, unsigned char* textons,
    unsigned char* blocks_img, int block_size, int window_size, int nb_blocks) {
    int value = 0;
    int pos_t = k + j * block_size + i * block_size * block_size;
    for (int l = 0; l < window_size; ++l) {
        for (int m = 0; m < window_size; ++m) {
            int pos_j = j - window_size / 2 + l;
            int pos_k = k - window_size / 2 + m;

            if (pos_j == j && pos_k == k)
                continue;

            if (pos_j < 0 || pos_j >= block_size ||
                pos_k < 0 || pos_k >= block_size) {
                value = value << 1;
            } else {
                int pos_it = pos_k + pos_j * block_size + i * block_size * block_size;
                if (blocks_img[pos_it] >= blocks_img[pos_t])
                    value = value << 1 | 1;
                else
                    value = value << 1;
            }
        }
    }
    textons[pos_t] = value;
}

__device__ 
int get_value_texton(unsigned char* texton, int i) {
    return texton[i];
}

__global__
void compute_histogram_block_gpu(int* histogram, unsigned char* texton,
    int size_histogram, int nb_blocks) {
    int i = /*blockDim.x  * blockIdx.x*/ + threadIdx.x;
    if (i >= size_histogram)
        return;
    int cellValue = get_value_texton(texton + blockIdx.x * size_histogram, i);
    atomicAdd(&(histogram[cellValue + blockIdx.x * size_histogram]), 1);
}

__global__
void compute_shared_histogram_block_gpu(int* histogram, unsigned char* texton,
    int size_histogram, int nb_blocks) {
    extern __shared__ int local_histogram[];

    int i = threadIdx.x;
    if (i >= size_histogram)
        return;
    local_histogram[i] = 0;
    __syncthreads();

    int cellValue = get_value_texton(texton + blockIdx.x * size_histogram, i);
    atomicAdd(&(local_histogram[cellValue]), 1);
    __syncthreads();

    atomicAdd(&(histogram[i + blockIdx.x * size_histogram]), local_histogram[i]);
}
