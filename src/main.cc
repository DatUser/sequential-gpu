#include <iostream>

#include "image.hh"
#include "save.hh"

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
    blocks.compute_textons_blocks();

    // Step 3: Compute histogram
    blocks.compute_histogram_blocks();

    // Step 4: Concatenate histograms
    std::vector<unsigned char> hist = blocks.get_concatenated_histograms();
    save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

    std::vector<Block*> blocks_data = blocks.get_blocks();
    Block* data = blocks_data[0];
    std::cout << *data << '\n';
    return 0;
}
