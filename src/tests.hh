#pragma once

#include "image.hh"
#include "image-gpu.hh"

#include <iostream>

bool are_images_equal(unsigned char* img1, unsigned char* img2, int size1, int size2);

template<typename T>
bool are_array_equal(T img1, T img2, int size1, int size2) {
    std::cout << "HERE" << std::endl;
    if (size1 != size2)
        return false;

    for (int i = 0; i < size1; ++i) {
        if (img1[i] != img2[i]) {
            std::cout << "Element nb : " << i << " difference : " << static_cast<unsigned>(img1[i]) << " != " << static_cast<unsigned>(img2[i]) << std::endl;
            return false;
        }
    }

    return true;
}
