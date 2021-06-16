#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.hh"
#include "save.hh"
#include "blocks-gpu.hh"
#include "image-gpu.hh"
#include "tests.hh"
#include "timer.hh"

int main() {
    // LBP algorithm

    // Load img
    // GPU
    ImageGPU img_gpu("data/test.jpg");
    auto start_gray_padd_gpu = start_timer();
    img_gpu.to_gray();
    img_gpu.padd_image();
    auto duration_gray_padd_gpu = duration(start_gray_padd_gpu); 
    //img_gpu.save_gray_ppm("gray.ppm");
    //img_gpu.save_padded_gray_ppm("padded_gray.ppm");

    // CPU
    int patch_size = 16;
    Image img_cpu("data/test.jpg");
    auto start_gray_padd_cpu = start_timer();
    img_cpu.set_patch_size(patch_size);
    auto other_img = img_cpu.to_gray();
    auto padded_img = other_img.add_padding_row();
    auto padded_img2 = padded_img.add_padding_col();
    auto duration_gray_padd_cpu = duration(start_gray_padd_cpu); 

    display_times(duration_gray_padd_cpu, duration_gray_padd_gpu, "Gray and padding");

    // TESTS

    bool are_eq1 = are_array_equal<unsigned char *>(other_img.get_data(), img_gpu.get_gray_data(), other_img.get_size(), img_gpu.get_size());
    bool are_eq2 = are_array_equal<unsigned char *>(padded_img2.get_data(), img_gpu.get_padded_gray_data(), padded_img2.get_size(), img_gpu.get_padded_size());
    std::cout << "--------------\n";
    std::cout << "Gray Img test: " << std::boolalpha << are_eq1 << '\n';
    std::cout << "Padded Gray Img test: " << std::boolalpha << are_eq2 << '\n';


    // Step 1: Compute blocks / tiling
    int window_size = 3;
    Blocks blocks = padded_img2.to_blocks(window_size);

    // Step 2: Compute textons
    // GPU
    //std::cout << "GPU version\n\n";
    BlocksGPU blocks_gpu(blocks, window_size);
    auto start_texton_gpu = start_timer();
    blocks_gpu.compute_textons();
    auto duration_texton_gpu = duration(start_texton_gpu); 

    auto start_histo_gpu = start_timer();
    blocks_gpu.compute_histogram_blocks();
    auto duration_histo_gpu = duration(start_histo_gpu); 


    /*for (int i = 0; i < 8; i++)
      std::cout << (int) blocks_gpu.textons_device[i] << std::endl;*/

    // CPU
    //std::cout << "CPU version\n\n";
    auto start_texton_cpu = start_timer();
    blocks.compute_textons_blocks();
    auto duration_texton_cpu = duration(start_texton_cpu); 

    display_times(duration_texton_cpu, duration_texton_gpu, "Texton");

    // Step 3: Compute histogram
    auto start_histo_cpu = start_timer();
    blocks.compute_histogram_blocks();


    // Step 4: Concatenate histograms
    std::vector<int> hist = blocks.get_concatenated_histograms();

    auto duration_histo_cpu = duration(start_histo_cpu); 
    display_times(duration_histo_cpu, duration_histo_gpu, "Histogram");

    save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

    bool are_histo_eq = are_array_equal<int *>(blocks.get_concatenated_histograms().data(), blocks_gpu.histogram,
       blocks.get_concatenated_histograms_size(), blocks_gpu.get_concatenated_histograms_size());
    std::cout << "--------------\n";
    std::cout << "Concatenate test: " << std::boolalpha << are_histo_eq << '\n';

    std::vector<Block*> blocks_data = blocks.get_blocks();
    Block* data = blocks_data[0];
    //std::cout << *data << '\n';

    return 0;
}
