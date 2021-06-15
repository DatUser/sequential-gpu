#pragma once

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
    void save_gray_ppm(const char* path) const;

private:
    // gpu image data
    // shape=(height, width, nb_channels)
    unsigned char* data;

    // gpu image gray data
    // shape=(height, width)
    unsigned char* gray_data;

    /* Attributes */
    // width / height of the image
    // number of channels of the image (r/g/b)
    int width;
    int height;
    int nb_channels;

    // how to divide the image
    int patch_size;
};

__global__
void to_gray_gpu(unsigned char* data);