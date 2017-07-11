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

__host__ __device__  void doPottsStep(float* arrayToUpdate, uint32_t* arrJ, float* arrP, float* m, float* s, float* w,
                                      float gamma, uint32_t row, uint32_t col, uint32_t width, uint32_t height, uint32_t nc,
                                      uint32_t y, uint32_t n, uint32_t length, uint32_t colorOffset, uint32_t upper, uint32_t lower, uint32_t chunkSize) {

    float d = 0;
    float p = 0;
    float dpg = 0;


//    uint32_t upper = min(chunkSize*(col+1), length);
//    uint32_t lower = chunkSize*col + 1;
//


    float mTemp, wDiffTemp;
    for(uint32_t r = lower; r <= upper; r++) {
        arrP[r - 1 + y*n] = s[r + y*(n+1)] - (normQuad(m, r + y*(n+1), nc, colorOffset) / w[r + y*(n+1)]);
        arrJ[(r - 1 + y*n)*2] = lower-1;
        arrJ[(r - 1 + y*n)*2+1] = upper;
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
                arrJ[(r - 1 + y*n)*2] = l - 1;
            }
        }
    }

    uint32_t r = upper;
    uint32_t l = arrJ[(r - 1 + y*n)*2];

//    if(row == 94) {
//        for(uint32_t j = lower; j <= upper; ++j) {
//            printf("row: %i col : %i j: %i r: %i l: %i\n", row, col, j, arrJ[(j - 1 + y*n)*2+1],arrJ[(j - 1 + y*n)*2]);
//        }
//    }

    while(r > lower) {
        for(uint32_t j = l; j < r; j++) {
            arrJ[(j + y*n)*2] = l;
            arrJ[(j + y*n)*2+1] = r;
        }
        if(row == 129) {
            printf("row: %i col: %i r: %i l: %i\n", row, col, r, l);
        }
        if (1 == l%chunkSize || length == 1) {
            printf("l mod 50 %i l: %i r: %i\n", l%2, l, r);
            arrJ[(l - 1 + y*n)*2+1] = r;
        }
        r = l;
        if (r < lower) break;
        l = arrJ[(r - 1 + y*n)*2];
    }

    if(row == 129) {
        for(uint32_t j = lower; j <= upper; ++j) {
            printf("row: %i col : %i j: %i r: %i l: %i\n", row, col, j, arrJ[(j - 1 + y*n)*2+1],arrJ[(j - 1 + y*n)*2]);
        }
    }

}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                                 uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize) {
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t col = threadIdx.y + blockDim.y * blockIdx.y;


    if(row < h && ((col*chunkSize) < w)) {
//        printf("row: %i col: %i\n", row, col);
        uint32_t y = row;
        uint32_t length = w;
        uint32_t upper = min(chunkSize*(col+1), length);
        uint32_t lower = chunkSize*col + 1;
        copyDataHorizontally(u, weights, m, s, wPotts, row, w, h, nc, colorOffset);
        doPottsStep(u, arrJ, arrP, m, s, wPotts, gamma, row, col, w, h, nc, y, nPotts, length, colorOffset, upper, lower, chunkSize);
    }
}

__global__ void copyHorizontally(float* u, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc, uint32_t colorOffset) {
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

__global__ void copyVertically(float* v, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc, uint32_t colorOffset) {
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
//        printf("col: %i row: %i upper: %i lower: %i length: %i\n", col, row, upper, lower, length);
        copyDataDiagonallyUpper(w_, weights, m, s, wPotts, col, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, col, nPotts, length, colorOffset, upper, lower, chunkSize);
    } else if (col > w && col < w+h) {
        uint32_t length = min(h - (col - w), w);
        uint32_t upper = min(chunkSize*(row+1), length);
        uint32_t lower = chunkSize*row + 1;
//        printf("col: %i row: %i upper: %i lower: %i length: %i\n", row, col-w, upper, lower, length);
        copyDataDiagonallyLower(w_, weights, m, s, wPotts, col-w, w, h, nc, colorOffset, nPotts, length);
        doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, col-1, nPotts, length, colorOffset, upper, lower, chunkSize);
    }
}

__global__ void copyDiagonallyUpper(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
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

__global__ void copyDiagonallyLower(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
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

__global__ void copyAntiDiagonallyUpper(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
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

__global__ void copyAntiDiagonallyLower(float* w_, uint32_t* arrJ, float* m, float* wPotts, uint32_t w, uint32_t h, uint32_t nc,
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