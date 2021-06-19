#pragma once

#include <iostream>

#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

float* to_float_ptr(int* vec, int size);