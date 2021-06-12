#include "image.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <cstdio>

Image::Image(int w, int h, int nb_chan, int p_size) :
    width(w),
    height(h),
    nb_channels(nb_chan),
    patch_size(p_size)
{
    data = new unsigned char[w * h * nb_chan];
}

Image::Image(const char* path) :
    patch_size(16)
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

void Image::save_ppm(const char* path) const {
    std::ofstream ofs(path, std::ios_base::out | std::ios_base::binary);
    ofs << "P6" << std::endl << width << ' ' << height << std::endl << "255" << std::endl;

    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            ofs << (char) data[j * width + i] << (char) data[j * width + i]  << (char) data[j * width + i];

    ofs.close();
}

Image Image::to_gray() const {
    Image img(width, height, 1, patch_size);

    for (int i = 0; i < width * height; ++i) {
        img.data[i] = data[i * 3] * 0.2989 + data[i * 3 + 1] * 0.5870 + data[i * 3 + 2] * 0.1140;
    }

    return img;
}

Image Image::add_padding_row() const {
    int new_height = height + patch_size - height % patch_size;

    Image new_img(width, new_height, 1, patch_size);
    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i >= height) {
                new_img.data[i * width + j] = data[(height-1) * width + j];
            } else {
                new_img.data[i * width + j] = data[i * width + j];
            }
        }
    }

    return new_img;
}

Image Image::add_padding_col() const {
    int new_width = width + patch_size - width % patch_size;

    Image new_img(new_width, height, 1, patch_size);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            if (j >= width) {
                new_img.data[i * new_width + j] = data[i * width + (width-1)];
            } else {
                new_img.data[i * new_width + j] = data[i * width + j];
            }
        }
    }

    return new_img;
}

Blocks Image::to_blocks(int window_size) const {
    Blocks blocks(height / patch_size, width / patch_size, patch_size);

    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            blocks.add_block(get_block(i, j, window_size));
        }
    }

    Block* tmp = (Block*) malloc(blocks.get_blocks_size() * sizeof(Block));
    for (int i = 0; i < blocks.get_blocks_size(); i++)
      tmp[i] = *blocks.get_hblocks()[i];


    cudaMemcpy(blocks.get_blocks(), tmp,
	blocks.get_blocks_size() * sizeof(Block), cudaMemcpyHostToDevice);

    free(tmp);

    return blocks;
}

Block* Image::get_block(int i, int j, int window_size) const {
    // Get block of size: patch_size * patch_size
    Block* block = new Block(patch_size, window_size);
    for (int k = 0; k < patch_size; ++k) {
        for (int l = 0; l < patch_size; ++ l) {
            unsigned char v = data[(i + k) * width + j + l];
            block->set_at(k, l, v);
        }
    }

    return block;
}
