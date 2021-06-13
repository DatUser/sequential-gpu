#include "blocks-gpu.hh"

#include "block-gpu.hh"

#include <iostream>

BlocksGPU::BlocksGPU(Blocks blocks, int w_size) {
    block_size = blocks.get_blocks()[0]->get_block_size();
    int size = block_size * block_size;
    nb_blocks = blocks.get_nb_blocks();
    window_size = w_size;

    printf("Before alloc textons\n");
    cudaMallocManaged(&textons_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    printf("Before alloc blocks\n");
    cudaMalloc(&blocks_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    printf("Before alloc loop\n");
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
    int nb_blocks_cuda = 64;
    //dim3 threads_(nb_blocks, block_size, block_size);
    printf("%d\n", (nb_blocks_cuda + nb_blocks) / nb_blocks_cuda);
    printf("%d\n", nb_blocks);
    dim3 threads_(256, block_size, block_size);
    dim3 blocks_(4 , 1, 1);

    //for (int i = 0; i < nb_blocks; ++i) {
    compute_texton_block_gpu<<<blocks_, threads_>>>(textons_device,
                                                    blocks_device,
                                                    block_size,
                                                    window_size,
                                                    nb_blocks);
    cudaCheckError();
   // }

    cudaDeviceSynchronize();
    cudaCheckError();
}
