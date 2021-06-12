#include <iostream>
#include <stdio.h>

#include "image.hh"
#include "save.hh"

__global__
void print_kernel() {
  printf("Hello from block %d, thread %d\n",blockIdx.x, threadIdx.x);
}

__global__
void compute_histogram_blocks_gpu(Blocks& blocks)
{
    int i = blockIdx.x * blockDim.x +threadIdx.x;
    if (i < blocks.get_blocks_size())
      blocks.compute_histogram_blocks_gpu();
}

int main() {
    // LBP algorithm
    print_kernel<<<2, 3>>>();
    cudaDeviceSynchronize();

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
    blocks.compute_textons_blocks();

    // Step 3: Compute histogram
    //blocks.compute_histogram_blocks_gpu();
    compute_histogram_blocks_gpu<<<blocks.get_blocks_size(), 1>>>(blocks);
    cudaDeviceSynchronize();
    cudaMemcpy(blocks.get_hblocks(), blocks.get_blocks(),
	blocks.get_blocks_size(), cudaMemcpyDeviceToHost);

    // Step 4: Concatenate histograms
    std::vector<unsigned char> hist = blocks.get_concatenated_histograms();
    save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

    Block** blocks_data = blocks.get_hblocks();
    Block* data = blocks_data[0];
    std::cout << *data << '\n';
    return 0;
}
