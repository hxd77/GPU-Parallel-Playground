// error.cuh
#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
{                                                                     \
    const cudaError_t error = call;                                   \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        fprintf(stderr, "CUDA error: %s:%d, ", __FILE__, __LINE__);   \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                      \
    }                                                                 \
}
