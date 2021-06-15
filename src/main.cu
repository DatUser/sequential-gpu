#include <iostream>
#include <chrono>

//#include "image.hh"
#include "save.hh"
//#include "gpu/block-gpu.hh"
#include "blocks-gpu.hh"
#include "image-gpu.hh"

int main() {
    // LBP algorithm

    // Load img
    ImageGPU img("data/test.jpg");
    img.to_gray();
    img.padd_image();
    img.save_gray_ppm("gray.ppm");
    img.save_padded_gray_ppm("padded_gray.ppm");


    // Set patch size
    /*int patch_size = 16;
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
    //std::cout << "GPU version\n\n";
    BlocksGPU blocks_gpu(blocks, window_size);
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    blocks_gpu.compute_textons();
    auto t2_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> ms_gpu = t2_gpu - t1_gpu;*/

    /*for (int i = 0; i < 8; i++)
      std::cout << (int) blocks_gpu.textons_device[i] << std::endl;*/

    // CPU
    //std::cout << "CPU version\n\n";
/*    auto t1_cpu = std::chrono::high_resolution_clock::now();
    blocks.compute_textons_blocks();
    auto t2_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> ms_cpu = t2_cpu - t1_cpu;

    std::cout << "CPU  Values" << std::endl;*/
    /*for (int i = 0; i < 8; i++)
      std::cout << (int) blocks.get_blocks()[0]->get_texton_at(0, i) << std::endl;*/
/*
    std::cout << "GPU excution time:\n" << ms_gpu.count() * 1000 << "ms\n\n";
    std::cout << "CPU execution time:\n" << ms_cpu.count() * 1000 << "ms\n";

    // Step 3: Compute histogram
    blocks.compute_histogram_blocks();

    // Step 4: Concatenate histograms
    std::vector<unsigned char> hist = blocks.get_concatenated_histograms();
    save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

    std::vector<Block*> blocks_data = blocks.get_blocks();
    Block* data = blocks_data[0];*/
    //std::cout << *data << '\n';

    return 0;
}
