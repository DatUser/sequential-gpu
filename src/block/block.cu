#include <bitset>
#include <iomanip>

#include "block.hh"

Block::Block(int b_size, int w_size) :
    block_size(b_size),
    window_size(w_size)
{
    block = new unsigned char[block_size * block_size];
    texton = new unsigned char[block_size * block_size];
    histogram = std::vector<unsigned int>(block_size * block_size, 0);
}

Block::~Block() {
    delete []block;
    delete []texton;
}

void Block::compute_texton_block() {
    int texton_idx = 0;
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            compute_pixel_texton(i, j, texton_idx);
        }
    }
}

void Block::compute_pixel_texton(int i, int j, int& idx) {
    int value = 0;
    for (int k = 0; k < window_size; ++k) {
        for (int l = 0; l < window_size; ++l) {
            int pos_i = i - window_size / 2 + k;
            int pos_j = j - window_size / 2 + l;

            if (pos_i == i && pos_j == j)
                continue;

            if (pos_i < 0 || pos_i >= block_size ||
                pos_j < 0 || pos_j >= block_size) {
                value = value << 1;
            } else {
                if (this->get_val_at(pos_i, pos_j) >= this->get_val_at(i, j))
                    value = value << 1 | 1;
                else
                    value = value << 1;
            }
        }
    }
    texton[idx] = value;
    ++idx;
}

void Block::compute_histogram_block() {
    for (int i = 0; i < block_size * block_size; ++i) {
        ++histogram[texton[i]];
    }
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
    os << "Block\n";
    auto b_size = block.get_block_size();
    for (int i = 0; i < b_size; ++i) {
        for (int j = 0; j < b_size; ++j) {
            os << (int) block.get_val_at(i, j) << ' ';
        }
        os << '\n';
    }

    os << "-------\nTextons:\n";
    os << "Bits (first 3)\n";
    for (int i = 0; i < 3; ++i) {
        int texton = block.get_texton_at(0, i);
        os << "At (0, " << i << "): "
            << std::bitset<8>(texton).to_string() << ": "
            << texton << '\n';
    }


    for (int i = 0; i < b_size; ++i) {
        for (int j = 0; j < b_size; ++j) {
            os << std::setw(3) << (int) block.get_texton_at(i, j) << ' ';
        }
        os << '\n';
    }
    os << "-------\nHistogram (first 10):\n";
    const auto histogram = block.get_histogram();
    for (int i = 0; i < 10; ++i) {
        os << "hist[" << i << "] = " << histogram[i] << '\n';
    }

    return os;
}
