#pragma once

#include "blocks.hh"

class Image {
    public:
        Image(int width, int height, int nb_channels);
        Image(const char* path);
        ~Image();

        /* Methods */
        Image to_gray() const;
        Image add_padding_row() const;
        Image add_padding_col() const;

        Blocks to_blocks() const;

        void save_ppm(const char* path) const;

        /* Setters */
        void set_patch_size(int p) { patch_size = p; }

        /* Attributes */
        unsigned char* data;
    private:
        Block get_block(int i, int j) const;

        /* Attributes */
        // width / height of the image
        // number of channels of the image (r/g/b)
        int width;
        int height;
        int nb_channels;

        // how to divide the image
        int patch_size;
};
