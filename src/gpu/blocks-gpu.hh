#pragma once

#include "blocks.hh"

#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

class BlocksGPU {
public:
    BlocksGPU(Blocks blocks, int window_size);
    ~BlocksGPU();

    // compute all the textons (each texton is computed on GPU)
    void compute_textons();
    void compute_histogram_blocks();

    // Data on the GPU
    // 3 dim tensor
    // shape=(nb_blocks, block_size, block_size)
    unsigned char* textons_device;
    unsigned char* blocks_device;
    // shape = (nb_blocks, block_size * block_size), it contains number up to 255
    int* histogram;

    // number of blocks in our image
    int nb_blocks;

    // size of one block (should be 16)
    int block_size;

    // window for the compute texton
    int window_size;
};
