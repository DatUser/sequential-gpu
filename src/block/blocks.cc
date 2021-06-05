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
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        compute_textons_block(blocks[i]);
    }
}

void Blocks::compute_textons_block(const Block* block) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            // Compute
        }
    }
}

void Blocks::compute_pixel_texton(Block block, int i, int j) {
    // TODO: maybe add window size in constructor
    int window_size = 3;

    for (int k = 0; k < window_size; ++k) {
        for (int l = 0; l < window_size; ++l) {
            int pos_i = i - window_size / 2 + k;
            int pos_j = j - window_size / 2 + l;

            if (pos_i < 0 || pos_i >= block_size ||
                pos_j < 0 || pos_j >= block_size) {
                // TODO
            }

            // TODO
        }
    }
}
