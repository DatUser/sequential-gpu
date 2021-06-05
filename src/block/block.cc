#include "block.hh"

Block::Block(int b_size) :
    block_size(b_size)
{
    block = new unsigned char[block_size * block_size];
}

/*Block::Block(int b_size, unsigned char* b) :
    block_size(b_size),
    block(b)
{
}*/

Block::~Block() {
    delete []block;

    for (unsigned int i = 0; i < textons.size(); ++i) {
        delete []textons[i];
    }
}

void Block::add_texton(unsigned char* texton) {
    textons.push_back(texton);
}

std::ostream& operator<<(std::ostream& os, const Block block) {
    os << "Blocks\n";
    auto b_size = block.get_block_size();
    for (int i = 0; i < b_size; ++i) {
        for (int j = 0; j < b_size; ++j) {
            os << (int) block.get_val_at(i, j) << ' ';
        }
        os << '\n';
    }

    return os;
}
