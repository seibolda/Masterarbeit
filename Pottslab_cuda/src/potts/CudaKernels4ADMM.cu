#ifndef _CUDA_KERNELS_4ADMM_H_
#define _CUDA_KERNELS_4ADMM_H_

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#endif

#include <cstdint>

__global__ void prepareHorizontalPottsProblems4ADMM(float* in, float* u, float* v, float* weights, float* weightsPrime,
                                             float* lam, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        u[index] = (weights[weightsIndex] * in[index] + v[index] * mu - lam[index]) / weightsPrime[weightsIndex];

    }
}

__global__ void prepareVerticalPottsProblems4ADMM(float* in, float* u, float* v, float* weights, float* weightsPrime,
                                             float* lam, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        v[index] = (weights[weightsIndex] * in[index] + u[index] * mu + lam[index]) / weightsPrime[weightsIndex];

    }
}

__global__ void updateLagrangeMultiplierKernel4ADMM(float* u, float* v, float* lam, float* temp, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        temp[index] = u[index] - v[index];
        lam[index] = lam[index] + temp[index] * mu;
    }
}

#endif