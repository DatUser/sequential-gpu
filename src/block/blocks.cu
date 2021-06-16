#include <cassert>

#include "blocks.hh"

Blocks::Blocks(int nb_r, int nb_c, int size) :
    nb_rows(nb_r),
    nb_cols(nb_c),
    block_size(size)
{
}

void Blocks::add_block(Block* b) {
    blocks.push_back(b);
}

void Blocks::compute_textons_blocks() {
    // Compute textons for each block
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        blocks[i]->compute_texton_block();
    }
}

void Blocks::compute_histogram_blocks() {
    // Compute histogram for each block
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        blocks[i]->compute_histogram_block();
    }
}

std::vector<int> Blocks::get_concatenated_histograms() {
    // Concatenate histograms
    std::vector<int> hist;
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        auto block_hist = blocks[i]->get_histogram();
        std::move(block_hist.begin(), block_hist.end(), std::back_inserter(hist));
    }

    // flatten array of shape (nb_tiles * blocks_size * block_size)
    assert(hist.size() == blocks.size() * block_size * block_size);
    return hist;
}
