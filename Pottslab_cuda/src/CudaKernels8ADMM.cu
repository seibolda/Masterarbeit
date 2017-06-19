#ifndef _CUDA_KERNELS_8ADMM_H_
#define _CUDA_KERNELS_8ADMM_H_

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#endif

#include <cstdint>

__global__ void prepareHorizontalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights,
                                                    float* weightsPrime, float* lam1, float* lam2, float* lam3, float mu,
                                                    uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        u[index] = (weights[weightsIndex] * in[index] + 2 * mu * (v[index] + w_[index] + z[index])
                    + 2 * (-lam1[index] - lam2[index] - lam3[index])) / weightsPrime[weightsIndex];

    }
}

__global__ void prepareDiagonalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights,
                                                  float* weightsPrime, float* lam2, float* lam4, float* lam6, float mu,
                                                  uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        w_[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + v[index] + z[index])
                     + 2 * (lam2[index] + lam4[index] - lam6[index])) / weightsPrime[weightsIndex];

    }
}

__global__ void prepareVerticalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights,
                                                    float* weightsPrime, float* lam1, float* lam4, float* lam5, float mu,
                                                    uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        v[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + w_[index] + z[index])
                    + 2 * (lam1[index] - lam4[index] - lam5[index])) / weightsPrime[weightsIndex];

    }
}

__global__ void prepareAntidiagonalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights,
                                                    float* weightsPrime, float* lam3, float* lam5, float* lam6, float mu,
                                                    uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        z[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + v[index] + w_[index])
                    + 2 * (lam3[index] + lam5[index] + lam6[index])) / weightsPrime[weightsIndex];

    }
}

__global__ void updateLagrangeMultiplierKernel8ADMM(float* u, float* v, float* w_, float* z, float* lam1, float* lam2, float* lam3,
                                                    float* lam4, float* lam5, float* lam6, float* temp, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        temp[index] = u[index] - v[index];
        lam1[index] = lam1[index] + mu * (u[index] - u[index]);
        lam2[index] = lam2[index] + mu * (u[index] - v[index]);
        lam3[index] = lam3[index] + mu * (u[index] - z[index]);
        lam4[index] = lam4[index] + mu * (v[index] - w_[index]);
        lam5[index] = lam5[index] + mu * (v[index] - z[index]);
        lam6[index] = lam6[index] + mu * (w_[index] - z[index]);
    }
}

#endif