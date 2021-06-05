#pragma once

#include <vector>
#include <iostream>

class Block {
    public:
        Block(int b_size, int w_size);
        ~Block();

        void add_texton(bool* texton);

        bool* get_texton(int i) const { return textons[i]; }
        unsigned int get_textons_size() const { return textons.size(); }
        int get_block_size() const { return block_size; }
        int get_window_size() const { return window_size; }
        unsigned char get_val_at(int i, int j) const { return block[i * block_size + j]; }

        void set_at(int i, int j, unsigned char val) { block[i * block_size + j] = val; }
    private:
        int block_size;
        int window_size;

        // flatten array of size (block_size * block_size)
        unsigned char* block;

        // flatten array of shape (block_size * block_size, window_size * window)size - 1)
        std::vector<bool*> textons;

        // size (block_size * block_size)
        std::vector<unsigned int> histogram;
};

std::ostream& operator<<(std::ostream& os, const Block block);
