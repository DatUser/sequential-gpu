#pragma once

#include "image-gpu.hh"
#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

class CanonicalGPU {
public:
  CanonicalGPU(const ImageGPU& image, int window_size);
  ~CanonicalGPU();

    // compute all the textons (each texton is computed on GPU)
    void compute_textons();
    void compute_shared_histogram_blocks();
    void compute_histogram_blocks();
    int get_concatenated_histograms_size() const {
        return nb_blocks * block_size * block_size;
    }

    // Data on the GPU
    // dimension as the padded image
    // shape (height, width)
    unsigned char* textons_device;
    unsigned char* padded_gray_data;
    // shape = (nb_blocks, block_size * block_size), it contains number up to 255
    int* histogram;

    int nb_blocks_x;
    int nb_blocks_y;
    // number of blocks in our image
    int nb_blocks;

    // size of one block (should be 16)
    int block_size;

    // window for the compute texton
    int window_size;
};
