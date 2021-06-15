#include "image-gpu.hh"

#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <cstdio>

__global__
void to_gray_gpu(unsigned char* gray_data, unsigned char* data, int width, int height, int nb_channels);

__global__
void padd_image_gpu(unsigned char* padded_gray_data, unsigned char* gray_data, int width, int height, int padded_width, int padded_height);

ImageGPU::ImageGPU(const char* path) {
    unsigned char* stbi_data = stbi_load(path, &width, &height, &nb_channels, 0);
    if (!stbi_data) {
        std::cout << "FAILURE to load the image: " << path << '\n';
        return;
    }

    patch_size = 16;

    int size = width * height * nb_channels;

    // compute padded width / height
    padded_width = width + patch_size - width % patch_size;
    padded_height =  height + patch_size - height % patch_size;

    // make allocations
    cudaMallocManaged(&data, sizeof(unsigned char) * size);
    cudaCheckError();

    cudaMallocManaged(&gray_data, sizeof(unsigned char) * width * height);
    cudaCheckError();

    cudaMallocManaged(&padded_gray_data, sizeof(unsigned char) * padded_width * padded_height);
    cudaCheckError();

    // copy the data to GPU
    cudaMemcpy(data, stbi_data, sizeof(unsigned char) * size, cudaMemcpyHostToDevice);
    cudaCheckError();

    // free the stbi data
    stbi_image_free(stbi_data);
}

ImageGPU::~ImageGPU() {
    cudaFree(data);
    cudaFree(gray_data);
    cudaFree(padded_gray_data);
}

void ImageGPU::to_gray() {
    int nb_blocks_x = 50;
    int nb_blocks_y = 50;
    dim3 blocks_(nb_blocks_x, nb_blocks_y);
    dim3 threads_((height + nb_blocks_x) / nb_blocks_x, (width + nb_blocks_y) / nb_blocks_y);
    to_gray_gpu<<<blocks_, threads_>>>(gray_data, data, width, height, nb_channels);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

void ImageGPU::padd_image() {
    int nb_blocks_x = 50;
    int nb_blocks_y = 50;
    dim3 blocks_(nb_blocks_x, nb_blocks_y);
    dim3 threads_((height + nb_blocks_x) / nb_blocks_x, (width + nb_blocks_y) / nb_blocks_y);
    padd_image_gpu<<<blocks_, threads_>>>(padded_gray_data, gray_data, width, height, padded_width, padded_height);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

void ImageGPU::save_gray_ppm(const char* path) const {
    std::ofstream ofs(path, std::ios_base::out | std::ios_base::binary);
    ofs << "P6" << std::endl << width << ' ' << height << std::endl << "255" << std::endl;

    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            ofs << (char) gray_data[j * width + i]
                << (char) gray_data[j * width + i]
                << (char) gray_data[j * width + i];

    ofs.close();
}

void ImageGPU::save_padded_gray_ppm(const char* path) const {
    std::ofstream ofs(path, std::ios_base::out | std::ios_base::binary);
    ofs << "P6" << std::endl << padded_width << ' ' << padded_height << std::endl << "255" << std::endl;

    for (int j = 0; j < padded_height; ++j)
        for (int i = 0; i < padded_width; ++i)
            ofs << (char) padded_gray_data[j * padded_width + i]
                << (char) padded_gray_data[j * padded_width + i]
                << (char) padded_gray_data[j * padded_width + i];

    ofs.close();
}

// -------------
// GPU functions
// -------------

__global__
void to_gray_gpu(unsigned char* gray_data, unsigned char* data, int width, int height, int nb_channels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < height && j < width) {
        // get r / g / b
        float r = (float) data[i * width * nb_channels + j * nb_channels];
        float g = (float) data[i * width * nb_channels + j * nb_channels + 1];
        float b = (float) data[i * width * nb_channels + j * nb_channels + 2];

        // to gray
        float pixel_intensity = r * 0.2989 + g * 0.5870 + b * 0.1140;
        gray_data[i * width + j] = (unsigned char) pixel_intensity;
    }
}

__global__
void padd_image_gpu(unsigned char* padded_gray_data,
                    unsigned char* gray_data,
                    int width, int height,
                    int padded_width, int padded_height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < padded_height && j < padded_width) {
        if (i < height && j < width) {
            padded_gray_data[i * padded_width + j] = gray_data[i * width + j];
        } else if (i < height && j >= width) {
            padded_gray_data[i * padded_width + j] = gray_data[i * width + width - 1];
        } else if (i >= height && j < width) {
            padded_gray_data[i * padded_width + j] = gray_data[(height - 1) * width + j];
        } else {
            padded_gray_data[i * padded_width + j] = gray_data[(height - 1) * width + width - 1];
        }
    }
}