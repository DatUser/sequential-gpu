#include "utils.hh"

__global__ void to_float_ptr_gpu(float* result, int* vec, int size);

float* to_float_ptr(int* vec, int size) {
    float* result;
    cudaMallocManaged(&result, sizeof(float) * size);
    cudaCheckError();

    int nb_blocks = 500;
    dim3 threads_((size + nb_blocks) / nb_blocks);
    dim3 blocks_(nb_blocks);
    to_float_ptr_gpu<<<blocks_, threads_>>>(result, vec, size);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    return result;
}

__global__ void to_float_ptr_gpu(float* result, int* vec, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    result[idx] = (float) vec[idx];
}