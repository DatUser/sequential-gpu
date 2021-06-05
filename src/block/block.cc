#include "block.hh"

Block::Block(int nb_r, int nb_c, int size) :
    nb_rows(nb_r),
    nb_cols(nb_c),
    block_size(size),
    nb_neighbors(8)
{
}

Block::~Block() {
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        delete []blocks[i];
    }

    for (unsigned int i = 0; i < textons.size(); ++i) {
        delete []textons[i];
    }
}

void Block::add_block(unsigned char* b) {
    blocks.push_back(b);
}

void Block::compute_textons() {
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        compute_texton(blocks[i]);

    }
}

void Block::compute_texton(const unsigned char* block) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            //auto texton = new unsigned char[nb_neighbors];

            (void) block;
            // TODO


        }
    }
}
