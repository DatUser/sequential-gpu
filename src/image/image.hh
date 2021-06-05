#pragma once

#include "block.hh"

class Image {
    public:
        Image(int width, int height, int nb_channels);
        Image(const char* path);
        ~Image();

        void save_ppm(const char* path) const;

        Image to_gray() const;
        Image add_padding_row();
        Image add_padding_col();

        Block to_blocks() const;

        void set_data(unsigned char* data) { this->data = data; }

    private:
        int width;
        int height;
        int nb_channels;

        unsigned char* data;
};
