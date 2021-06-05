#include "image.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <cstdio>

Image::Image(int w, int h, int nb_chan) :
    width(w),
    height(h),
    nb_channels(nb_chan)
{
}

Image::Image(const char* path)
{
    data = stbi_load(path, &width, &height, &nb_channels, 0);
    if (!data) {
        std::cout << "FAILURE to load the image: " << path << '\n';
    }

}

Image::~Image() {
    if (data) {
        stbi_image_free(data);
    }
}

void Image::save_ppm(const char* path) {
    std::ofstream ofs(path, std::ios_base::out | std::ios_base::binary);
    ofs << "P6" << std::endl << width << ' ' << height << std::endl << "255" << std::endl;

    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            ofs << (char) data[j * width + i] << (char) data[j * width + i]  << (char) data[j * width + i] ;       // red, green, blue

    ofs.close();
}

Image Image::to_gray() {
    Image img(width, height, 1);

    auto new_data = new unsigned char[width * height];

    for (int i = 0; i < width * height; ++i) {
        new_data[i] = data[i * 3] * 0.2989 + data[i * 3 + 1] * 0.5870 + data[i * 3 + 2] * 0.1140;
    }

    img.set_data(new_data);
    return img;
}

Image Image::add_padding_row() {
    int new_height = height + 16 - height % 16;

    Image new_img(width, new_height, 1);
    auto new_data = new unsigned char[width * new_height];

    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i >= height) {
                new_data[i * width + j] = data[(height-1) * width + j];
            } else {
                new_data[i * width + j] = data[i * width + j];
            }
        }
    }

    new_img.set_data(new_data);
    return new_img;
}

Image Image::add_padding_col() {
    int new_width = width + 16 - width % 16;

    Image new_img(new_width, height, 1);
    auto new_data = new unsigned char[new_width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            if (j >= width) {
                new_data[i * new_width + j] = data[i * width + (width-1)];
            } else {
                new_data[i * new_width + j] = data[i * width + j];
            }
        }
    }

    new_img.set_data(new_data);
    return new_img;
}
