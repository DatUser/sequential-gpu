#include <iostream>

#include "image.hh"

int main() {
    // Load img
    Image img("data/test.jpg");

    // Img to grayscale
    auto other_img = img.to_gray();
    other_img.save_ppm("data/test_gray.ppm");

    // Add padding row & col
    auto padded_img = other_img.add_padding_row();
    auto padded_img2 = padded_img.add_padding_col();
    padded_img2.save_ppm("data/padded_test.ppm");

    // Compute blocks
    Block blocks = padded_img2.to_blocks();

    auto blocks_data = blocks.get_blocks();
    auto data = blocks_data[0];
    for (int i = 0; i < 16; ++i) {
        std::cout << int(data[i]) << '\n';
    }
    return 0;
}
