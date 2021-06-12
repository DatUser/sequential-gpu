#include <cassert>

#include "blocks.hh"

__host__
Blocks::Blocks(int nb_r, int nb_c, int size) :
    nb_rows(nb_r),
    nb_cols(nb_c),
    block_size(size),
    blocks_size(0)
{
  h_blocks = (Block**) malloc(nb_rows * nb_cols * sizeof(Block*));
  cudaMalloc(&blocks, nb_rows * nb_cols * sizeof(Block));
}

/*void Blocks::add_block(Block* b) {
    blocks.push_back(b);
}*/

__host__
void Blocks::add_block(Block* b) {
  h_blocks[blocks_size] = b;
  blocks_size++;
}

__host__
void Blocks::compute_textons_blocks() {
    // Compute textons for each block
    unsigned int size = nb_rows * nb_cols;
    for (unsigned int i = 0; i < size; ++i) {
        h_blocks[i]->compute_texton_block();
    }
}

__host__
void Blocks::compute_histogram_blocks() {
    // Compute histogram for each block
    unsigned int size = nb_rows * nb_cols;
    for (unsigned int i = 0; i < size; ++i) {
        h_blocks[i]->compute_histogram_block();
    }
}

__device__
void Blocks::compute_histogram_blocks_gpu()
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  int size = nb_rows * nb_cols;
  if (i < size)
    blocks[i].compute_histogram_block_gpu();
}

std::vector<unsigned char> Blocks::get_concatenated_histograms() {
    // Concatenate histograms
    std::vector<unsigned char> hist;
    unsigned int size = nb_rows * nb_cols;
    for (unsigned int i = 0; i < size; ++i) {
        auto block_hist = h_blocks[i]->get_h_histogram();
        std::move(block_hist.begin(), block_hist.end(), std::back_inserter(hist));
    }

    // flatten array of shape (nb_tiles * blocks_size * block_size)
    assert(hist.size() == size * block_size * block_size);
    return hist;
}
