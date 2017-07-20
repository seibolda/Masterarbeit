#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#include <cstdint>
#include "potts/CudaPotts.cu"

__global__ void setWeightsKernel(float* weights, uint32_t w, uint32_t h) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        setWeights(weights, index);
    }
}

__global__ void updateWeightsPrimeKernel(float* weightsPrime, float* weights, uint32_t w, uint32_t h, float mu, float factor) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        updateWeightsPrime(weightsPrime, weights, mu, factor, index);
    }
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                                 uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                 uint32_t chunkSizeOffset) {
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t col = threadIdx.y + blockDim.y * blockIdx.y;


    if(row < h) {
        applyHorizontalPottsSolver(u, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    }
}

__global__ void copyBackHorizontally(float* u, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc, uint32_t colorOffset) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        copyDataBackHorizontally(u, arrJ, m, wPotts, y, x, c, w, h, colorOffset);
    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                               uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                               uint32_t chunkSizeOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        applyVerticalPottsSolver(v, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    }
}

__global__ void copyBackVertically(float* v, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc, uint32_t colorOffset) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(x < w && y < h && c < nc) {
        copyDataBackVertically(v, arrJ, m, wPotts, y, x, c, w, h, colorOffset);
    }
}

__global__ void applyDiagonalPottsSolverKernel(float* w_, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                               uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                               uint32_t chunkSizeOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        applyDiagonalUpperPottsSolver(w_, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    } else if (col > w && col < w+h) {
        applyDiagonalLowerPottsSolver(w_, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    }
}

__global__ void copyBackDiagonallyUpper(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
                                        uint32_t colorOffset, uint32_t nPotts) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(col < w && row < h && c < nc) {
        uint32_t length = min(h, w - col);
        if(row < length) {
            copyDataBackDiagonallyUpper(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffset, nPotts);
        }
    }
}

__global__ void copyBackDiagonallyLower(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
                                        uint32_t colorOffset, uint32_t nPotts) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(col < w && row < h && c < nc & row > 0) {
        uint32_t length = min(h - row, w);
        if(col < length) {
            copyDataBackDiagonallyLower(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffset, nPotts);
        }
    }
}

__global__ void applyAntiDiagonalPottsSolverKernel(float* z, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                                   uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                   uint32_t chunkSizeOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        applyAntiDiagonalUpperPottsSolver(z, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    } else if (col > w && col < w+h) {
        applyAntiDiagonalLowerPottsSolver(z, weights, arrJ, arrP, m, s, wPotts, gamma, w, h, nc, nPotts, colorOffset, chunkSize, chunkSizeOffset, row, col);
    }
}

__global__ void copyBackAntiDiagonallyUpper(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
                                        uint32_t colorOffset, uint32_t nPotts) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(col < w && row < h && c < nc) {
        uint32_t length = min(h, w - col);
        if(row < length) {
            copyDataBackAntiDiagonallyUpper(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffset, nPotts);
        }
    }
}

__global__ void copyBackAntiDiagonallyLower(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
                                        uint32_t colorOffset, uint32_t nPotts) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

    if(col < w && row < h && c < nc & row > 0) {
        uint32_t length = min(h - row, w);
        if(col < length) {
            copyDataBackAntiDiagonallyLower(w_, arrJ, m, wPotts, row, col, c, w, h, colorOffset, nPotts);
        }
    }
}






#endif