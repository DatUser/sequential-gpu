#include "tests.hh"


bool are_images_equal(unsigned char* img1, unsigned char* img2, int size1, int size2) {
    if (size1 != size2)
        return false;

    for (int i = 0; i < size1; ++i) {
        if (img1[i] != img2[i])
            return false;
    }

    return true;
}