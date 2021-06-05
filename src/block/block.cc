#include "block.hh"

Block::Block(int b_size, int w_size) :
    block_size(b_size),
    window_size(w_size)
{
    block = new unsigned char[block_size * block_size];
}

Block::~Block() {
    delete []block;

    for (unsigned int i = 0; i < textons.size(); ++i) {
        delete []textons[i];
    }
}

void Block::add_texton(bool* texton) {
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

    os << "-------\nTextons (3 first):\n";

    int window_size = block.get_window_size();
    int nb_neigh = window_size * window_size - 1;
    unsigned int nb_textons = block.get_textons_size();
    os << "Nb Textons: " << nb_textons << '\n';

    for (unsigned int i = 0; i < 3 && i < nb_textons; ++i) {
        bool* texton = block.get_texton(i);

        for (int j = 0; j < nb_neigh; ++j) {
            os << (int) texton[j] << ' ';
        }
        os << '\n';
    }

    return os;
}
