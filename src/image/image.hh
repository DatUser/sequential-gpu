#pragma once

#include "block.hh"

class Image {
    public:
        Image(int width, int height, int nb_channels);
        Image(const char* path);
        ~Image();

        /* Methods */
        Image to_gray() const;
        Image add_padding_row() const;
        Image add_padding_col() const;

        Block to_blocks() const;

        void save_ppm(const char* path) const;

        /* Setters */
        void set_patch_size(int p) { patch_size = p; }

        /* Attributes */
        unsigned char* data;
    private:
        unsigned char* get_block(int i, int j) const;

        /* Attributes */
        int width;
        int height;
        int nb_channels;

        int patch_size;
};
