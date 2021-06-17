#include "canonanical-gpu.hh"

#include <iostream>

CanonicalGPU::CanonicalGPU(ImageGPU image, int window_size) {
	this->padded_gray_data = image.get_padded_gray_data();
    this->block_size = image.get_patch_size();
    this->nb_blocks_x = image.get_nb_blocks_x();
    this->nb_blocks_y = image.get_nb_blocks_y();
    this->nb_blocks = nb_blocks_x * nb_blocks_y;

    int size = block_size * block_size;
    this->window_size = window_size;

    cudaMallocManaged(&textons_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    cudaMallocManaged(&histogram, sizeof(int) * nb_blocks * size);
    cudaCheckError();

    cudaMemset(histogram, 0, sizeof(int) * nb_blocks * size);
    cudaCheckError();
    
}

CanonicalGPU::~CanonicalGPU() {
    cudaFree(textons_device);
    cudaFree(histogram);
}

__device__
int get_image_index(int x, int y, int nb_blocks_x) {
    return x + blockIdx.x * blockDim.x + y * blockDim.x * nb_blocks_x;
}

__device__ 
unsigned char get_padded_gray_data_value(unsigned char* padded_gray_data, int x, int y, int nb_blocks_x) {
    return padded_gray_data[get_image_index(x, y, nb_blocks_x)];
}

__global__ void compute_texton_block_canonical_gpu(unsigned char* textons, unsigned char* padded_gray_data, int block_size, int window_radius, int nb_blocks_x) {
    unsigned char value = 0;
    int x = threadsIdx.x;
    int y = threadIdx.y;

    unsigned char central_pixel = get_padded_gray_data_value(padded_gray_data, x, y, nb_blocks_x);

    int idx = 0;
    for (int i = -window_radius; i <= window_radius; ++i) {
        int x_i = x + i;
        if (x_i < 0 || x_i >= block_size) {
            ++idx;
            continue;
        }
        for (int j = -window_radius; j <= window_radius; ++j) {
            int y_j = y + j;
            if (y_j < 0 || y_j >= block_size) {
                ++idx;
                continue;
            }
            if (i == 0 && j == 0)
                continue;
            if (central_pixel >= get_padded_gray_data_value(x_i, y_j))
                value |= 1 << (8 - idx);
            ++idx;
        }
    }
    textons[get_image_index(x, y, nb_blocks_x)] = value;
}

void CanonicalGPU::compute_textons() {
    dim3 blocks_(nb_blocks_x, nb_blocks_y);
    dim3 threads_(block_size, block_size);

    compute_texton_block_canonical_gpu<<<blocks_, threads_>>>(textons_device,
                                                    padded_gray_data,
                                                    block_size,
                                                    window_size);
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError();
}

void CanonicalGPU::compute_histogram_blocks() {
    int size = block_size * block_size;
    //for (int i = 0; i < nb_blocks; ++i) {
    dim3 threads_(size);
    dim3 blocks_(nb_blocks);

    compute_histogram_block_gpu<<<blocks_, threads_>>>(histogram /*+ i * size*/
    , textons_device /*+ i * size*/, size, nb_blocks);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
    //}

}

void CanonicalGPU::compute_shared_histogram_blocks() {
    int size = block_size * block_size;
    dim3 threads_(size);
    dim3 blocks_(nb_blocks);

    compute_shared_histogram_block_gpu<<<blocks_, threads_, size * sizeof(int)>>>(histogram, textons_device, size, nb_blocks);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

