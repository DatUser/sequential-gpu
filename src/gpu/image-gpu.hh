#pragma once

#include "blocks-gpu.hh"

#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

class ImageGPU {
public:
    ImageGPU(const char* path);
    ~ImageGPU();

    void to_gray();
    void padd_image();
    void save_gray_ppm(const char* path) const;
    void save_padded_gray_ppm(const char* path) const;
    BlocksGPU to_blocks(int window_size) const;

    /* getters */
    unsigned char* get_gray_data() const { return gray_data; }
    unsigned char* get_padded_gray_data() const { return padded_gray_data; }
    int get_padded_width() const { return padded_width; }
    void set_padded_gray_data(unsigned char* padded_gray_data) {
        cudaMemcpy(this->padded_gray_data, padded_gray_data, sizeof(unsigned char) * get_padded_size(), cudaMemcpyDefault);
        //this->padded_gray_data = padded_gray_data;
    }
  int get_size() const { return width * height; }
    int get_padded_size() const { return padded_width * padded_height; }
    int get_patch_size() const { return patch_size; }
    int get_nb_blocks_x() const { return padded_width / patch_size; }
    int get_nb_blocks_y() const { return padded_height / patch_size; }

private:
    // gpu image data
    // shape=(height, width, nb_channels)
    unsigned char* data;

    // gpu image gray data
    // shape=(height, width)
    unsigned char* gray_data;

    // gpu padded image
    unsigned char* padded_gray_data;

    /* Attributes */
    // width / height of the image
    // number of channels of the image (r/g/b)
    int width;
    int padded_width;
    int padded_height;
    int height;
    int nb_channels;

    // how to divide the image
    int patch_size;
};

__global__
void compute_blocks_device(int window_size, unsigned char* blocks_device,
                           unsigned char* padded_gray_data, int p_size, int nb_tiles_x, int padded_width,
                           int padded_height, int patch_size);
