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

void Blocks::compute_textons_block(Block* block) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            compute_pixel_texton(block, i, j);
        }
    }
}

void Blocks::compute_pixel_texton(Block* block, int i, int j) {
    // TODO: maybe add window size in constructor
    int window_size = 3;
    bool* texton = new bool[window_size * window_size - 1];

    int texton_idx = 0;
    for (int k = 0; k < window_size; ++k) {
        for (int l = 0; l < window_size; ++l) {
            int pos_i = i - window_size / 2 + k;
            int pos_j = j - window_size / 2 + l;

            if (pos_i == i && pos_j == j)
                continue;

            if (pos_i < 0 || pos_i >= block_size ||
                pos_j < 0 || pos_j >= block_size) {
                texton[texton_idx] = false;
            } else {
                if (block->get_val_at(pos_i, pos_j) >= block->get_val_at(i, j))
                    texton[texton_idx] = true;
                else
                    texton[texton_idx] = false;
            }
            texton_idx++;
        }
    }
    block->add_texton(texton);
}