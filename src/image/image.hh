#pragma once

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

#include "blocks.hh"

class Image {
    public:
        Image(int width, int height, int nb_channels, int patch_size);
        Image(const char* path);
        ~Image();

        /* Methods */

        Image to_gray() const;
        Image add_padding_row() const;
        Image add_padding_col() const;


        Blocks to_blocks(int window_size) const;

        void save_ppm(const char* path) const;

        /* Getters */
        unsigned char* get_data() const { return data; }
        int get_size() const { return width * height; }

        /* Setters */
        void set_patch_size(int p) { patch_size = p; }
    private:
        Block* get_block(int i, int j, int window_size) const;

        /* Attributes */
        // width / height of the image
        // number of channels of the image (r/g/b)
        int width;
        int height;
        int nb_channels;
        unsigned char* data;

        // how to divide the image
        int patch_size;
};
