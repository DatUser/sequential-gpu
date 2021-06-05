#include <iostream>

#include "image.hh"

int main() {
    auto img = Image("data/test.jpg");
    auto other_img = img.to_gray();

    other_img.save_ppm("data/test_gray.ppm");
    return 0;
}
