#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#include <cstdint>
#include "potts/CudaPotts.cu"
#include "potts/CudaCopyData.cu"

__global__ void setWeightsKernel(float* weights, uint32_t w, uint32_t h) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        weights[index] = 1;
    }
}

__global__ void updateWeightsPrimeKernel(float* weightsPrime, float* weights, uint32_t w, uint32_t h, float mu, float factor) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < w && y < h) {
        uint32_t index = x + w * y;
        weightsPrime[index] = weights[index] + (factor * mu);
    }
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                                 uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize) {
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t col = threadIdx.y + blockDim.y * blockIdx.y;


    if(row < h && ((col*chunkSize) < w)) {
        uint32_t y = row;
        uint32_t length = w;
        uint32_t upper = min(chunkSize*(col+1), length);
        uint32_t lower = chunkSize*col + 1;
        copyDataHorizontally(u, weights, m, s, wPotts, row, w, h, nc, colorOffset);
        doPottsStep(u, arrJ, arrP, m, s, wPotts, gamma, row, col, w, h, nc, y, nPotts, length, colorOffset, upper, lower, chunkSize);
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
                                               uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        uint32_t y = col;
        uint32_t length = h;
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
        copyDataVertically(v, weights, m, s, wPotts, col, w, h, nc, colorOffset);
        doPottsStep(v, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, y, nPotts, length, colorOffset, upper, lower, chunkSize);
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
                                               uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        uint32_t length = min(h, w - col);
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
        copyDataDiagonallyUpper(w_, weights, m, s, wPotts, col, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, col, nPotts, length, colorOffset, upper, lower, chunkSize);
    } else if (col > w && col < w+h) {
        uint32_t length = min(h - (col - w), w);
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
        copyDataDiagonallyLower(w_, weights, m, s, wPotts, col-w, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, col-1, nPotts, length, colorOffset, upper, lower, chunkSize);
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
                                                   uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < w) {
        uint32_t length = min(h, w - col);
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
        copyDataAntiDiagonallyUpper(z, weights, m, s, wPotts, col, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(z, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, col, nPotts, length, colorOffset, upper, lower, chunkSize);
    } else if (col > w && col < w+h) {
        uint32_t length = min(h - (col - w), w);
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
        copyDataAntiDiagonallyLower(z, weights, m, s, wPotts, col-w, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(z, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, col-1, nPotts, length, colorOffset, upper, lower, chunkSize);
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