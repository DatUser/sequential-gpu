#pragma once

#include <vector>

class Block {
    public:
        Block(int nb_r, int nb_c);
        ~Block();

        void add_block(unsigned char* b);

        std::vector<unsigned char*> get_blocks() const { return blocks; }
    private:
        int nb_rows;
        int nb_cols;

        std::vector<unsigned char*> blocks;
};
