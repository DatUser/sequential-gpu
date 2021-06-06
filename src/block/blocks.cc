#include "blocks.hh"

#include <unordered_map>

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
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        compute_histogram_block(blocks[i]);
    }
}

void Blocks::compute_histogram_block(Block *block) {
    // TODO
    // map id to position
    //std::unordered_map<std::string, int> umap;

    // for i in  256
    // [] -> ""
    // { "": i }
    // block.vec[umap[""]]
}
