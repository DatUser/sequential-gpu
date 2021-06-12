#pragma once

#include <cstdio>

__global__
void compute_texton_block_gpu(unsigned char*  texton, unsigned char* block_img,
    int block_size, int window_size);

__device__
void compute_pixel_texton_gpu(int i, int j, unsigned char* texton,
    unsigned char* block_img, int block_size, int window_size);
