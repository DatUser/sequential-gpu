#include <iostream>

#include "image.hh"
#include "save.hh"
#include "gpu/block-gpu.hh"

#define cudaCheckError() {                                                                       \
        cudaError_t e=cudaGetLastError();                                                        \
        if(e!=cudaSuccess) {                                                                     \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    }

int main() {
    // LBP algorithm

    // Load img
    Image img("data/test.jpg");

    // Set patch size
    int patch_size = 16;
    img.set_patch_size(patch_size);

    // Img to grayscale
    auto other_img = img.to_gray();
    other_img.save_ppm("data/test_gray.ppm");

    // Add padding row & col
    auto padded_img = other_img.add_padding_row();
    auto padded_img2 = padded_img.add_padding_col();
    padded_img2.save_ppm("data/padded_test.ppm");

    // Step 1: Compute blocks / tiling
    int window_size = 3;
    Blocks blocks = padded_img2.to_blocks(window_size);

    // Step 2: Compute textons
    // GPU
    int block_size = blocks.get_blocks()[0]->get_block_size();
    int size = block_size * block_size;
    unsigned char** textons_device = 
      (unsigned char**) malloc(sizeof(unsigned char*) * blocks.get_nb_blocks());
    unsigned char** blocks_device = 
      (unsigned char**) malloc(sizeof(unsigned char*) * blocks.get_nb_blocks());
    for (int i = 0; i < blocks.get_nb_blocks(); i++)
    {
      //unsigned char* texton_device = malloc(blocks.get_blocks_size());
      cudaMallocManaged(&textons_device[i], size * sizeof(unsigned char));
      cudaCheckError();
      cudaMalloc(&blocks_device[i], size * sizeof(unsigned char));
      cudaCheckError();
      cudaMemcpy(blocks_device[i], blocks.get_blocks()[i]->get_block(),
	  16 * 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
      cudaCheckError();
    }

    dim3 threads_(block_size, block_size);
    dim3 blocks_(1, 1);

    for (int i = 0; i < 8; i++)
      std::cout << (int) textons_device[0][i] << std::endl;

    std::cout << std::endl;

    for (int i = 0; i < /*blocks.get_nb_blocks()*/1; i++)
    {
	compute_texton_block_gpu<<<blocks_, threads_>>>(textons_device[i],
	    blocks_device[i], block_size, window_size);
        cudaCheckError();
    }

    cudaDeviceSynchronize();
    cudaCheckError();

    blocks.compute_textons_blocks();//CPU

    for (int i = 0; i < 8; i++)
      std::cout << (int) textons_device[0][i] << std::endl;

    // Step 3: Compute histogram
    blocks.compute_histogram_blocks();

    // Step 4: Concatenate histograms
    std::vector<unsigned char> hist = blocks.get_concatenated_histograms();
    save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

    std::vector<Block*> blocks_data = blocks.get_blocks();
    Block* data = blocks_data[0];
    std::cout << *data << '\n';

    for (int i = 0; i < blocks.get_nb_blocks(); i++)
    {
      cudaFree(textons_device[i]);
      cudaCheckError();
      cudaFree(blocks_device[i]);
      cudaCheckError();
    }

    free(textons_device);
    free(blocks_device);

    return 0;
}
