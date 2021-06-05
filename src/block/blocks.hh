#pragma once

#include <vector>

#include "block.hh"

class Blocks {
    public:
        Blocks(int nb_r, int nb_c, int size);
        ~Blocks() = default;

        void add_block(Block* b);
        void compute_textons_blocks();

        /* Getters */
        std::vector<Block*> get_blocks() const { return blocks; }

    private:
        /* Methods */
        void compute_textons_block(Block* block);
        void compute_pixel_texton(Block* block, int i, int j);

        /* Attributes*/
        // Number of blocks row-wise and column-wise
        int nb_rows;
        int nb_cols;

        // size of one block
        // size of the window
        int block_size;
        int window_size;

        std::vector<Block*> blocks;

};
