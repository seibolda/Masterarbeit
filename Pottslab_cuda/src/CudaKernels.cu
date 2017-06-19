#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#define HORIZONTAL 0
#define VERTICAL 1
#define DIAGONAL 2
#define ANTIDIAGONAL 3

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
                                  float gamma, uint32_t rowCol, uint32_t n, uint32_t widthOrHeight, uint32_t nc, uint8_t direction) {
    uint32_t y = rowCol;
    uint32_t h = widthOrHeight;

    float d = 0;
    float p = 0;
    float dpg = 0;
    float mu1, mu2, mu3;


    switch (direction) {
        case HORIZONTAL:
            copyDataHorizontally(arrayToUpdate, weights, m, s, w, y, n, h, nc);
            break;
        case VERTICAL:
            copyDataVertically(arrayToUpdate, weights, m, s, w, y, h, n, nc);
            break;
    }

    float wTemp, mTemp, wDiffTemp;
    for(uint32_t r = 1; r <= n; r++) {
        arrP[r - 1 + y*n] = s[r + y*(n+1)] - (normQuad(m, r + y*(n+1), nc, (n+1)*(h+1)) / w[r + y*(n+1)]);;
        arrJ[r - 1 + y*n] = 0;
        for(uint32_t l = r; l >= 2; l--) {
            mTemp = 0;
            for(uint32_t k = 0; k < nc; k++) {
                mTemp = mTemp + (m[r + y*(n+1) + k*(n+1)*(h+1)]-m[l - 1 + y*(n+1) + k*(n+1)*(h+1)]) * (m[r + y*(n+1) + k*(n+1)*(h+1)]-m[l - 1 + y*(n+1) + k*(n+1)*(h+1)]);
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

    uint32_t r = n;
    uint32_t l = arrJ[r - 1 + y*n];
    while(r > 0) {
        wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
        mu1 = (m[r + y*(n+1)] - m[l + y*(n+1)]) / wTemp;
        if(nc > 1) {
            mu2 = (m[r + y*(n+1) + (n+1)*(h+1)] - m[l + y*(n+1) + (n+1)*(h+1)]) / wTemp;
            mu3 = (m[r + y*(n+1) + 2*(n+1)*(h+1)] - m[l + y*(n+1) + 2*(n+1)*(h+1)]) / wTemp;
        }

        switch (direction) {
            case HORIZONTAL:
                copyDataBackHorizontally(arrayToUpdate, l, r, mu1, mu2, mu3, y, n, h, nc);
                break;
            case VERTICAL:
                copyDataBackVertically(arrayToUpdate, l, r, mu1, mu2, mu3, y, h, n, nc);
                break;
        }
        /*for(uint32_t j = l; j < r; j++) {
            arrayToUpdate[j + y*n] = mu1;
            if(nc > 1) {
                arrayToUpdate[j + y * n + n * h] = mu2;
                arrayToUpdate[j + y * n + 2 * n * h] = mu3;
            }
        }*/
        r = l;
        if (r < 1) break;
        l = arrJ[r - 1 + y*n];
    }
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                                 float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(y < h) {
        doPottsStep(u, weights, arrJ, arrP, m, s, wPotts, gamma, y, w, h, nc, HORIZONTAL);
    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                               float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if(x < w) {
        doPottsStep(v, weights, arrJ, arrP, m, s, wPotts, gamma, x, h, w, nc, VERTICAL);
    }
}



#endif