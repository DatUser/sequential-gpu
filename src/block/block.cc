#include "block.hh"

Block::Block(int nb_r, int nb_c) :
    nb_rows(nb_r),
    nb_cols(nb_c)
{
}

Block::~Block() {
    for (unsigned int i = 0; i < blocks.size(); ++i) {
        delete []blocks[i];
    }
}

void Block::add_block(unsigned char* b) {
    blocks.push_back(b);
}
