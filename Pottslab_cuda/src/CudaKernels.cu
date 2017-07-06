#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#define HORIZONTAL 0
#define VERTICAL 1
#define DIAGONAL_UPPER 2
#define DIAGONAL_LOWER 3
#define ANTIDIAGONAL_UPPER 4
#define ANTIDIAGONAL_LOWER 5

#include <cstdint>
#include "CudaCopyData.cu"

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

__host__ __device__  void doPottsStep(float* arrayToUpdate, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* w,
                                      float gamma, uint32_t row, uint32_t col, uint32_t width, uint32_t height, uint32_t nc, uint8_t direction,
                                      uint32_t y, uint32_t n, uint32_t length, uint32_t colorOffset) {

    float d = 0;
    float p = 0;
    float dpg = 0;
    float mu1, mu2, mu3;


    float wTemp, mTemp, wDiffTemp;
    for(uint32_t r = 1; r <= length; r++) {
        arrP[r - 1 + y*n] = s[r + y*(n+1)] - (normQuad(m, r + y*(n+1), nc, colorOffset) / w[r + y*(n+1)]);
        arrJ[r - 1 + y*n] = 0;
        for(uint32_t l = r; l >= 2; l--) {
            mTemp = 0;
            for(uint8_t k = 0; k < nc; k++) {
                mTemp = mTemp + (m[r + y*(n+1) + k*colorOffset]-m[l - 1 + y*(n+1) + k*colorOffset]) * (m[r + y*(n+1) + k*colorOffset]-m[l - 1 + y*(n+1) + k*colorOffset]);
            }
            wDiffTemp = w[r + y*(n+1)] - w[l - 1 + y*(n+1)];
            if(0 == wDiffTemp) {
                d = 0;
            } else {
                d = s[r + y*(n+1)] - s[l - 1 + y*(n+1)] - (mTemp / wDiffTemp);
            }
            dpg = d + gamma;
            if(dpg > arrP[r-1 + y*n]) {
                break;
            }
            p = arrP[l - 2 + y*n] + dpg;
            if(p < arrP[r - 1 + y*n]) {
                arrP[r - 1 + y*n] = p;
                arrJ[r - 1 + y*n] = l - 1;
            }
        }
    }

    uint32_t r = length;
    uint32_t l = arrJ[r - 1 + y*n];
    while(r > 0) {
        wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
        mu1 = (m[r + y*(n+1)] - m[l + y*(n+1)]) / wTemp;
        if(nc > 1) {
            mu2 = (m[r + y*(n+1) + colorOffset] - m[l + y*(n+1) + colorOffset]) / wTemp;
            mu3 = (m[r + y*(n+1) + 2*colorOffset] - m[l + y*(n+1) + 2*colorOffset]) / wTemp;
        }

        switch (direction) {
            case HORIZONTAL:
                copyDataBackHorizontally(arrayToUpdate, l, r, mu1, mu2, mu3, row, width, height, nc);
                break;
            case VERTICAL:
                copyDataBackVertically(arrayToUpdate, l, r, mu1, mu2, mu3, col, width, height, nc);
                break;
            case DIAGONAL_UPPER:
                copyDataBackDiagonallyUpper(arrayToUpdate, l, r, mu1, mu2, mu3, col, width, height, nc);
                break;
            case DIAGONAL_LOWER:
                copyDataBackDiagonallyLower(arrayToUpdate, l, r, mu1, mu2, mu3, row, width, height, nc);
                break;
            case ANTIDIAGONAL_UPPER:
                copyDataBackAntiDiagonallyUpper(arrayToUpdate, l, r, mu1, mu2, mu3, col, width, height, nc);
                break;
            case ANTIDIAGONAL_LOWER:
                copyDataBackAntiDiagonallyLower(arrayToUpdate, l, r, mu1, mu2, mu3, row, width, height, nc);
                break;
        }
        r = l;
        if (r < 1) break;
        l = arrJ[r - 1 + y*n];
    }
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                                 float gamma, uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset) {
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < h) {
        uint32_t y = row;
        uint32_t length = w;
        copyDataHorizontally(u, weights, m, s, wPotts, row, w, h, nc, colorOffset);
        doPottsStep(u, weights, arrJ, arrP, m, s, wPotts, gamma, row, 0, w, h, nc, HORIZONTAL, y, nPotts, length, colorOffset);
    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                               float gamma, uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if(col < w) {
        uint32_t y = col;
        uint32_t length = h;
        copyDataVertically(v, weights, m, s, wPotts, col, w, h, nc, colorOffset);
        doPottsStep(v, weights, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, VERTICAL, y, nPotts, length, colorOffset);
    }
}

__global__ void applyDiagonalPottsSolverKernel(float* w_, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                                    float gamma, uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if(col < w) {
        uint32_t length = min(h, w - col);
        copyDataDiagonallyUpper(w_, weights, m, s, wPotts, col, w, h, nc, colorOffset);
        doPottsStep(w_, weights, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, DIAGONAL_UPPER, col, nPotts, length, colorOffset);
    } else if (col > w && col < w+h) {
        uint32_t length = min(h - (col - w), w);
        copyDataDiagonallyLower(w_, weights, m, s, wPotts, col-w, w, h, nc, colorOffset);
        doPottsStep(w_, weights, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, DIAGONAL_LOWER, col-1, nPotts, length, colorOffset);
    }
}

__global__ void applyAntiDiagonalPottsSolverKernel(float* z, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                                        float gamma, uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset) {
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if(col < w) {
        uint32_t length = min(h, w - col);
        copyDataAntiDiagonallyUpper(z, weights, m, s, wPotts, col, w, h, nc, colorOffset);
        doPottsStep(z, weights, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, ANTIDIAGONAL_UPPER, col, nPotts, length, colorOffset);
    } else if (col > w && col < w+h) {
        uint32_t length = min(h - (col - w), w);
        copyDataAntiDiagonallyLower(z, weights, m, s, wPotts, col-w, w, h, nc, colorOffset);
        doPottsStep(z, weights, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, ANTIDIAGONAL_LOWER, col-1, nPotts, length, colorOffset);
    }
}








#endif