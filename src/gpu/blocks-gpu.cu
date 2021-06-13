#include "blocks-gpu.hh"

#include "block-gpu.hh"

#include <iostream>

BlocksGPU::BlocksGPU(Blocks blocks, int w_size) {
    block_size = blocks.get_blocks()[0]->get_block_size();
    int size = block_size * block_size;
    nb_blocks = blocks.get_nb_blocks();
    window_size = w_size;

    textons_device =
        (unsigned char**) malloc(sizeof(unsigned char*) * nb_blocks);

    blocks_device = 
        (unsigned char**) malloc(sizeof(unsigned char*) * nb_blocks);

    for (int i = 0; i < nb_blocks; ++i) {
        // allocate textons on device
        cudaMallocManaged(&textons_device[i], size * sizeof(unsigned char));
        cudaCheckError();

        // allocate blocks on device
        cudaMalloc(&blocks_device[i], size * sizeof(unsigned char));
        cudaCheckError();

        // copy the data on the device
        cudaMemcpy(blocks_device[i],
                   blocks.get_blocks()[i]->get_block(),
                   block_size * block_size * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        cudaCheckError();
    }
}

BlocksGPU::~BlocksGPU() {
    for (int i = 0; i < nb_blocks; ++i) {
        cudaFree(textons_device[i]);
        cudaCheckError();

        cudaFree(blocks_device[i]);
        cudaCheckError();
    }

    free(textons_device);
    free(blocks_device);
}

void BlocksGPU::compute_textons() {
    dim3 threads_(block_size, block_size);
    dim3 blocks_(1, 1);

    for (int i = 0; i < nb_blocks; ++i) {
        compute_texton_block_gpu<<<blocks_, threads_>>>(textons_device[i],
                                                        blocks_device[i],
                                                        block_size,
                                                        window_size);
        cudaCheckError();
    }

    cudaDeviceSynchronize();
    cudaCheckError();
}
