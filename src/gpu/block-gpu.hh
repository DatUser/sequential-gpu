#pragma once

#include <cstdio>

__global__
void compute_texton_block_gpu(unsigned char*  texton, unsigned char* block_img,
    int block_size, int window_size, int nb_blocks);

__device__
void compute_pixel_texton_gpu(int i, int j, int k, unsigned char* texton,
    unsigned char* block_img, int block_size, int window_size, int nb_blocks);

__global__
void compute_histogram_block_gpu(int* histogram, unsigned char* texton,
    int size_histogram, int nb_blocks);
