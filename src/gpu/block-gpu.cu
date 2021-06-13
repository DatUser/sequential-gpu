#include "block-gpu.hh"

__global__
void compute_texton_block_gpu(unsigned char* texton, unsigned char* block_img,
    int block_size, int window_size) {
    int i = blockDim.x *blockIdx.x + threadIdx.x;
    int j = blockDim.y *blockIdx.y + threadIdx.y;

    if (i * block_size + j < block_size * block_size)
    {
      compute_pixel_texton_gpu(i, j, texton, block_img,
                               block_size, window_size);
      //printf("%i\n", texton[texton_idx]);
      //printf("i: %i\nj: %i\nvalue:%i\n---\n", i, j, texton[i * block_size + j]);
    }
}

__device__
void compute_pixel_texton_gpu(int i, int j, unsigned char* texton,
    unsigned char* block_img, int block_size, int window_size) {
    int value = 0;
    for (int k = 0; k < window_size; ++k) {
        for (int l = 0; l < window_size; ++l) {
            int pos_i = i - window_size / 2 + k;
            int pos_j = j - window_size / 2 + l;

            if (pos_i == i && pos_j == j)
                continue;

            if (pos_i < 0 || pos_i >= block_size ||
                pos_j < 0 || pos_j >= block_size) {
                value = value << 1;
            } else {
                if (block_img[pos_i * block_size + pos_j]
		  >= block_img[i * block_size + j])
                    value = value << 1 | 1;
                else
                    value = value << 1;
            }
        }
    }
    texton[i * block_size + j] = value;
}
