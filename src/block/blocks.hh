#pragma once

#include <vector>

#include "block.hh"

class Blocks {
    public:
        __host__
	Blocks(int nb_r, int nb_c, int size);
	__host__
        ~Blocks() = default;

	__host__
        void add_block(Block* b);

	__host__
        void compute_textons_blocks();
	__host__
        void compute_histogram_blocks();

	__device__
        void compute_histogram_blocks_gpu();
        std::vector<unsigned char> get_concatenated_histograms();

        /* Getters */
        //std::vector<Block*> get_blocks() const { return blocks; }
	Block**  get_hblocks() const { return h_blocks; }
	Block*  get_blocks() const { return blocks; }
        int get_block_size() const { return block_size; }
	__host__ __device__
	int get_blocks_capacity() const { return nb_rows * nb_cols; }
	__host__ __device__
	int get_blocks_size() const { return blocks_size; }

    private:
        /* Attributes*/
        // Number of blocks row-wise and column-wise
        int nb_rows;
        int nb_cols;

        // size of one block
        int block_size;

        //std::vector<Block*> blocks;
	//Host blocks
	Block** h_blocks;
	//Device blocks
	Block* blocks;
	
	//Current number of stored blocks
	int blocks_size;
};
