#include "canonical-gpu.hh"
#include "image-gpu.hh"

#include <iostream>

  CanonicalGPU::CanonicalGPU(const ImageGPU& image, int window_size) {
	this->padded_gray_data = image.get_padded_gray_data();
    this->block_size = image.get_patch_size();
    this->nb_blocks_x = image.get_nb_blocks_x();
    this->nb_blocks_y = image.get_nb_blocks_y();
    this->nb_blocks = nb_blocks_x * nb_blocks_y;

    int size = block_size * block_size;
    this->window_size = window_size;

    cudaMallocManaged(&textons_device, sizeof(unsigned char) * nb_blocks * size);
    cudaCheckError();

    cudaMallocManaged(&histogram, sizeof(int) * nb_blocks * size);
    cudaCheckError();

    cudaMemset(histogram, 0, sizeof(int) * nb_blocks * size);
    cudaCheckError();
    
}

CanonicalGPU::~CanonicalGPU() {
    cudaFree(textons_device);
    cudaFree(histogram);
}

__device__
unsigned int get_image_index(unsigned int x, unsigned int y, int nb_blocks_x) {
    // return x_offset + y_offset + x_block_offset + y_block_offset;
  return x + y * (blockDim.x * nb_blocks_x) + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * (blockDim.x * nb_blocks_x);
}

__device__ 
unsigned char get_padded_gray_data_value(unsigned char* padded_gray_data, int x, int y, int nb_blocks_x) {
    return padded_gray_data[get_image_index(x, y, nb_blocks_x)];
}

__global__ void compute_texton_block_canonical_gpu(unsigned char* textons, unsigned char* padded_gray_data,
                                                   int block_size, int window_radius, int nb_blocks_x) {
    unsigned char value = 0;
    int x = threadIdx.x;
    int y = threadIdx.y;

    unsigned char central_pixel = get_padded_gray_data_value(padded_gray_data, x, y, nb_blocks_x);

    int idx = 7;

    for (int j = -window_radius; j <= window_radius; ++j) {
	int y_j = y + j;
        for (int i = -window_radius; i <= window_radius; ++i) {
            if (y_j < 0 || y_j >= block_size) {
              --idx;
              continue;
            }
	    int x_i = x + i;
            if (x_i < 0 || x_i >= block_size) {
                --idx;
                continue;
            }
            if (i == 0 && j == 0)
                continue;

            if (get_padded_gray_data_value(padded_gray_data, x_i, y_j, nb_blocks_x) >= central_pixel)
                value |= 1 << idx;
            --idx;
        }
    }
    unsigned image_index = get_image_index(x, y, nb_blocks_x);
  textons[get_image_index(x, y, nb_blocks_x)] = value;
}

void CanonicalGPU::compute_textons() {
    dim3 blocks_(nb_blocks_x, nb_blocks_y);
    dim3 threads_(block_size, block_size);
    int window_radius = window_size / 2;
    compute_texton_block_canonical_gpu<<<blocks_, threads_>>>(textons_device,
                                                    padded_gray_data,
                                                    block_size,
                                                    window_radius, nb_blocks_x);
    cudaCheckError();

    cudaDeviceSynchronize();
    std::cout << *textons_device;
    cudaCheckError();
}

__device__
unsigned char get_value_texton_canonical(unsigned char* texton, unsigned int i) {
  return texton[i];
}

__global__
void compute_shared_histogram_block_gpu_canonical(int* histogram, unsigned char* texton,
                                        int size_histogram, int nb_blocks_x) {
  extern __shared__ int local_histogram[];

  unsigned int i = threadIdx.x;
  unsigned int j = threadIdx.y;
  unsigned int index_1D = i + j * blockDim.x;
  unsigned int offset_local_histogram = i + j * nb_blocks_x * blockDim.x;
  if (index_1D >= size_histogram)
    return;
  local_histogram[index_1D] = 0;
  __syncthreads();

  unsigned int offset_x = blockDim.x * blockIdx.x;
  unsigned int offset_y = blockDim.y * blockIdx.y;
  unsigned int offset_total = offset_x + offset_y * nb_blocks_x * blockDim.x;
  unsigned char cellValue = get_value_texton_canonical(texton + offset_total, offset_local_histogram);
  atomicAdd(&(local_histogram[cellValue]), 1);
  __syncthreads();

  unsigned int index_histogram = blockIdx.x + blockIdx.y * nb_blocks_x;
  unsigned int offset_histogram = index_histogram * size_histogram;
  atomicAdd(&(histogram[index_1D + offset_histogram]), local_histogram[index_1D]);
}

__global__
void compute_histogram_block_gpu_canonical(int* histogram, unsigned char* texton,
                                        int size_histogram, int nb_blocks_x) {

  unsigned int i = threadIdx.x;
  unsigned int j = threadIdx.y;
  unsigned int offset_local_histogram = i + j * nb_blocks_x * blockDim.x;

  unsigned int offset_x = blockDim.x * blockIdx.x;
  unsigned int offset_y = blockDim.y * blockIdx.y;
  unsigned int offset_total = offset_x + offset_y * nb_blocks_x * blockDim.x;
  unsigned char cellValue = get_value_texton_canonical(texton + offset_total, offset_local_histogram);

  unsigned int index_histogram = blockIdx.x + blockIdx.y * nb_blocks_x;
  unsigned int offset_histogram = index_histogram * size_histogram;
  atomicAdd(&(histogram[cellValue + offset_histogram]), 1);
}

void CanonicalGPU::compute_shared_histogram_blocks() {
  dim3 blocks_(nb_blocks_x, nb_blocks_y);
  dim3 threads_(block_size, block_size);
  int size = block_size * block_size;
  compute_shared_histogram_block_gpu_canonical<<<blocks_, threads_, block_size * block_size * sizeof(int)>>>(histogram, textons_device,
                                                                                            size, nb_blocks_x);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

void CanonicalGPU::compute_histogram_blocks() {
  dim3 blocks_(nb_blocks_x, nb_blocks_y);
  dim3 threads_(block_size, block_size);
  int size = block_size * block_size;
  compute_histogram_block_gpu_canonical<<<blocks_, threads_>>>(histogram, textons_device,
                                                                      size, nb_blocks_x);

    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}