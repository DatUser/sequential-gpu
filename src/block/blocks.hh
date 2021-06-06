#pragma once

#include <vector>

#include "block.hh"

class Blocks {
    public:
        Blocks(int nb_r, int nb_c, int size);
        ~Blocks() = default;

        void add_block(Block* b);

        void compute_textons_blocks();
        void compute_histogram_blocks();
        std::vector<unsigned char> get_concatenated_histograms();

        /* Getters */
        std::vector<Block*> get_blocks() const { return blocks; }
        int get_block_size() const { return block_size; }

    private:
        /* Attributes*/
        // Number of blocks row-wise and column-wise
        int nb_rows;
        int nb_cols;

        // size of one block
        int block_size;

        std::vector<Block*> blocks;

};
