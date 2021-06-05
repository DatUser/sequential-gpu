#pragma once

#include <vector>

class Block {
    public:
        Block(int nb_r, int nb_c, int size);
        ~Block();

        void add_block(unsigned char* b);
        void compute_textons();

        /* Getters */
        std::vector<unsigned char*> get_blocks() const { return blocks; }

        /* Setters */
        void set_nb_neighbors(int nb) { nb_neighbors = nb; }

    private:
        /* Methods */
        void compute_texton(const unsigned char* block);

        /* Attributes*/
        int nb_rows;
        int nb_cols;

        int block_size;
        int nb_neighbors;

        std::vector<unsigned char*> blocks;
        std::vector<unsigned char*> textons;
};
