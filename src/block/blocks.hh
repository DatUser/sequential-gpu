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


        std::vector<int> get_concatenated_histograms();
        int get_concatenated_histograms_size() {
            return nb_rows * nb_cols * block_size * block_size;
        }

        /* Getters */
        std::vector<Block*> get_blocks() const { return blocks; }
        int get_block_size() const { return block_size; }
	int get_nb_blocks() const { return nb_rows * nb_cols; }

    private:
        /* Attributes*/
        // Number of blocks row-wise and column-wise
        int nb_rows;
        int nb_cols;

        // size of one block
        int block_size;

        std::vector<Block*> blocks;

};
