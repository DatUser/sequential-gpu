#include "blocks-gpu.hh"

#include "block-gpu.hh"

#include <iostream>

BlocksGPU::BlocksGPU(Blocks blocks, int w_size) {
    block_size = blocks.get_blocks()[0]->get_block_size();
    int size = block_size * block_size;
    nb_blocks = blocks.get_nb_blocks();
    window_size = w_size;

    cudaMallocManaged(&textons_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    cudaMalloc(&blocks_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    for (int i = 0; i < nb_blocks; ++i) {
        // copy the data on the device
        cudaMemcpy((blocks_device + i * size),
                   blocks.get_blocks()[i]->get_block(),
                   size * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        cudaCheckError();
    }
}

BlocksGPU::~BlocksGPU() {
    cudaFree(textons_device);
    cudaFree(blocks_device);
}

void BlocksGPU::compute_textons() {
    int nb_blocks_cuda_x = 4;
    dim3 threads_(nb_blocks_cuda_x, block_size, block_size);
    dim3 blocks_((nb_blocks + nb_blocks_cuda_x) / nb_blocks_cuda_x, 1, 1);

    compute_texton_block_gpu<<<blocks_, threads_>>>(textons_device,
                                                    blocks_device,
                                                    block_size,
                                                    window_size,
                                                    nb_blocks);
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError();
}
