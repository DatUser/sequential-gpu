#pragma once

#include <fstream>

inline void save_csv(const char* filepath, const char* sep,
        std::vector<unsigned char> data, unsigned int nb_cols) {
    std::ofstream csv_ofstream(filepath);

    for (unsigned int i = 0; i < data.size() / nb_cols; ++i) {
        for (unsigned int j = 0; j < nb_cols - 1; ++j) {
            csv_ofstream << (int) data[i * nb_cols + j] << sep;
        }
        csv_ofstream << (int) data[i * nb_cols + nb_cols - 1] << '\n';
    }

    csv_ofstream.close();
}
