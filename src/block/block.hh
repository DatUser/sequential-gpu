#pragma once

#include <vector>
#include <iostream>

class Block {
    public:
        Block(int b_size);
        Block(int b_size, unsigned char* b);
        ~Block();

        void add_texton(unsigned char* texton);

        int get_block_size() const { return block_size; }
        unsigned char get_val_at(int i, int j) const { return block[i * block_size + j]; }

        void set_at(int i, int j, unsigned char val) { block[i * block_size + j] = val; }
    private:
        int block_size;

        // flatten array of size (block_size * block_size)
        unsigned char* block;

        // flatten array of shape (block_size * block_size, window_size * window)size - 1)
        std::vector<unsigned char*> textons;
};

std::ostream& operator<<(std::ostream& os, const Block block);
