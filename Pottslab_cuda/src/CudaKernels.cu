#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#include <cstdint>

/*__global__ void copyTestKernel(float* in, float* out, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        out[index] = in[index];
    }
}*/

__global__ void printArrayKernel(float* array, uint32_t w, uint32_t h) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        if(0 != array[index])
            printf("Index: %d, Value: %f\n", index, array[index]);
    }
}

__global__ void setWeightsKernel(float* weights, uint32_t w, uint32_t h) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        weights[index] = 0.5;
    }
}

__global__ void updateWeightsPrimeKernel(float* weightsPrime, float* weights, uint32_t w, uint32_t h, float mu) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        weightsPrime[index] = weights[index] + mu;
    }
}

__device__ void prepareHorizontalAndVerticalPottsProblems(float* inputImage, float* targetImg, float* sourceImg,
                                                          float* weights, float* weightsPrime, float* lam, float mu,
                                                          uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        uint32_t weightsIndex = x + w * y;

        targetImg[index] = (weights[weightsIndex] * inputImage[index] + sourceImg[index] * mu - lam[index]) / weightsPrime[weightsIndex];;

    }
}

__global__ void prepareHorizontalPottsProblems(float* in, float* u, float* v, float* weights, float* weightsPrime,
                                               float* lam, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    prepareHorizontalAndVerticalPottsProblems(in, u, v, weights, weightsPrime, lam, mu, w, h, nc);
}

__global__ void prepareVerticalPottsProblems(float* in, float* u, float* v, float* weights, float* weightsPrime,
                                               float* lam, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    prepareHorizontalAndVerticalPottsProblems(in, v, u, weights, weightsPrime, lam, mu, w, h, nc);
}

__global__ void updateLagrangeMultiplierKernel(float* u, float* v, float* lam, float* temp, float mu, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        temp[index] = u[index] - v[index];
        lam[index] = lam[index] + temp[index] * mu;
    }
}

__global__ void updateErrorKernel(float* error, float* temp, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        uint32_t index = x + w * y + w * h * c;
        atomicAdd(&error[0], temp[index] * temp[index]);
    }
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(y < h) {

    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, float gamma, uint32_t w, uint32_t h, uint32_t nc) {

}


#endif