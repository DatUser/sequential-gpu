#pragma once

#include <vector>
#include <iostream>

class Block {
    public:
        Block(int b_size, int w_size);
        ~Block();

        void compute_texton_block();

        unsigned char get_val_at(int i, int j) const { return block[i * block_size + j]; }
        unsigned char get_texton_at(int i, int j) const { return texton[i * block_size + j]; }

        void set_at(int i, int j, unsigned char val) { block[i * block_size + j] = val; }

        /* Getters */
        int get_block_size() const { return block_size; }
        int get_window_size() const { return window_size; }

    private:
        /* Methods */
        void compute_pixel_texton(int i, int j, int& idx);

        /* Attributes */
        int block_size;
        int window_size;

        // flatten array of size (block_size * block_size)
        unsigned char* block;

        // flatten array of shape (block_size * block_size)
        unsigned char* texton;

        // size (block_size * block_size)
        std::vector<unsigned int> histogram;
};

std::ostream& operator<<(std::ostream& os, const Block& block);
