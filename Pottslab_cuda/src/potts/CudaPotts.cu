#ifndef _CUDA_POTTS_H_
#define _CUDA_POTTS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif

#include <cstdint>

__host__ __device__ void setWeights(float* weights, uint32_t index) {
    weights[index] = 1;
}



__host__ __device__ void updateWeightsPrime(float* weightsPrime, float* weights, float mu, float factor, uint32_t index) {
    weightsPrime[index] = weights[index] + (factor * mu);
}



__host__ __device__ float normQuad(float* array, uint32_t position, uint32_t nc, uint32_t colorOffset) {
    float result = 0;
    for(uint32_t c = 0; c < nc; c++) {
        result += array[position + c*colorOffset]*array[position + c*colorOffset];
    }
    return result;
}



__host__ __device__  void doPottsStep(float* arrayToUpdate, uint32_t* arrJ, float* arrP, float* m, float* s, float* w,
                                      float gamma, uint32_t row, uint32_t col, uint32_t width, uint32_t height, uint32_t nc,
                                      uint32_t y, uint32_t n, uint32_t length, uint32_t colorOffset, uint32_t upper, uint32_t lower,
                                      uint32_t chunkSize, uint32_t chunkSizeOffset) {

    float d = 0;
    float p = 0;
    float dpg = 0;

    float mTemp, wDiffTemp;

    for(uint32_t r = lower; r <= upper; r++) {
        arrP[r - 1 + y*n] = s[r + y*(n+1)] - (normQuad(m, r + y*(n+1), nc, colorOffset) / w[r + y*(n+1)]);
        arrJ[(r - 1 + y*n)*2] = lower-1;
        arrJ[(r - 1 + y*n)*2+1] = upper;
        for(uint32_t l = r; l >= lower + 1; l--) {
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

    while(r > lower-1) {
        for(uint32_t j = l; j < r; j++) {
            arrJ[(j + y*n)*2] = l;
            arrJ[(j + y*n)*2+1] = r;
        }
        if (1 == ((l+chunkSizeOffset)%chunkSize)) {
            arrJ[(l - 1 + y*n)*2+1] = r;
        }
        r = l;
        if (r < lower) break;
        l = arrJ[(r - 1 + y*n)*2];
    }

}



__host__ __device__ void copyDataHorizontally(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t row, uint32_t width,
                                              uint32_t height, uint32_t nc, uint32_t colorOffset) {
    float wTemp = 0;
    size_t idxBase = row*(width+1);
    for(uint32_t j = 0; j < width; j++) {

        wTemp = weights[j + row*width];

        for(uint32_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[j + row*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }

        float nQ = normQuad(arrayToUpdate, j+row*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackHorizontally(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                  uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset) {

    uint32_t y = row;
    uint32_t n = width;

    uint32_t r = arrJ[(col + y*n)*2+1];
    uint32_t l = arrJ[(col + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[col + row*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void copyDataVertically(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t col, uint32_t width, uint32_t height, uint32_t nc, uint32_t colorOffset) {
    float wTemp = 0;
    size_t idxBase = col*(height+1);
    for(uint32_t j = 0; j < height; j++) {

        wTemp = weights[col + j*width];
        for(uint32_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[col + j*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }
        float nQ = normQuad(arrayToUpdate, col + j*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackVertically(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset) {

    uint32_t y = col;
    uint32_t n = height;

    uint32_t r = arrJ[(row + y*n)*2+1];
    uint32_t l = arrJ[(row + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[col + row*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void copyDataDiagonallyUpper(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t col, uint32_t width,
                                                 uint32_t height, uint32_t nc, uint32_t colorOffset, uint32_t smallerDimension, uint32_t sDiag) {

    size_t idxBase = col*(smallerDimension+1);

    float wTemp = 0;
    for(uint32_t j = 0; j < sDiag; j++) {

        wTemp = weights[j+col + j*width];
        for(uint8_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[j+col + j*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }
        float nQ = normQuad(arrayToUpdate, j+col + j*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackDiagonallyUpper(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                     uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset, uint32_t n) {

    uint32_t y = col;

    uint32_t r = arrJ[(row + y*n)*2+1];
    uint32_t l = arrJ[(row + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[row + col + row*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void copyDataDiagonallyLower(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t row, uint32_t width,
                                                 uint32_t height, uint32_t nc, uint32_t colorOffset, uint32_t smallerDimension, uint32_t sDiag) {

    size_t idxBase = (row-1)*(smallerDimension+1) + width*(smallerDimension+1);

    float wTemp = 0;
    for(uint32_t j = 0; j < sDiag; j++) {

        wTemp = weights[j + row*width + j*width];
        for(uint8_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[j + row*width + j*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }
        float nQ = normQuad(arrayToUpdate, j + row*width + j*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackDiagonallyLower(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                     uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset, uint32_t n) {

    uint32_t y = row+width-1;

    uint32_t r = arrJ[(col + y*n)*2+1];
    uint32_t l = arrJ[(col + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[col + row*width + col*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void copyDataAntiDiagonallyUpper(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t col, uint32_t width,
                                                     uint32_t height, uint32_t nc, uint32_t colorOffset, uint32_t smallerDimension, uint32_t sDiag) {

    size_t idxBase = col*(smallerDimension+1);

    float wTemp = 0;
    for(uint32_t j = 0; j < sDiag; j++) {

        wTemp = weights[width-1-(col+j) + j*width];
        for(uint8_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[width-1-(col+j) + j*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }
        float nQ = normQuad(arrayToUpdate, width-1-(col+j) + j*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackAntiDiagonallyUpper(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                         uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset, uint32_t n) {

    uint32_t y = col;

    uint32_t r = arrJ[(row + y*n)*2+1];
    uint32_t l = arrJ[(row + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[width-1-(col+row) + row*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void copyDataAntiDiagonallyLower(float* arrayToUpdate, float* weights, float* m, float* s, float* w, uint32_t row, uint32_t width,
                                                     uint32_t height, uint32_t nc, uint32_t colorOffset, uint32_t smallerDimension, uint32_t sDiag) {

    size_t idxBase = (row-1)*(smallerDimension+1) + width*(smallerDimension+1);

    float wTemp = 0;
    for(uint32_t j = 0; j < sDiag; j++) {

        wTemp = weights[width-1-j + (j+row)*width];
        for(uint8_t c = 0; c < nc; c++) {
            m[j + 1 + idxBase + c*colorOffset] = arrayToUpdate[width-1-j + (j+row)*width + c*width*height] * wTemp + m[j + idxBase + c*colorOffset];
        }
        float nQ = normQuad(arrayToUpdate, width-1-j + (j+row)*width, nc, width*height);
        s[j + 1 + idxBase] = nQ * wTemp + s[j + idxBase];
        w[j + 1 + idxBase] = w[j + idxBase] + wTemp;
    }
}

__host__ __device__ void copyDataBackAntiDiagonallyLower(float* arrayToUpdate, uint32_t* arrJ, float* m, float* w, uint32_t row, uint32_t col,
                                                         uint32_t c, uint32_t width, uint32_t height, uint32_t colorOffset, uint32_t n) {

    uint32_t y = row+width-1;

    uint32_t r = arrJ[(col + y*n)*2+1];
    uint32_t l = arrJ[(col + y*n)*2];

    float wTemp = w[r + y*(n+1)] - w[l + y*(n+1)];
    arrayToUpdate[width-1-col + (col+row)*width + c*width*height] = (m[r + y*(n+1) + c*colorOffset] - m[l + y*(n+1) + c*colorOffset]) / wTemp;
}



__host__ __device__ void applyHorizontalPottsSolver(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                                    uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                    uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t y = row;
    uint32_t length = w;
    uint32_t upper = min(chunkSize*(col+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*col + 1, chunkSize*col + 1 - chunkSizeOffset);
    copyDataHorizontally(u, weights, m, s, wPotts, row, w, h, nc, colorOffset);
    doPottsStep(u, arrJ, arrP, m, s, wPotts, gamma, row, col, w, h, nc, y, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}



__host__ __device__ void applyVerticalPottsSolver(float* v, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,float gamma,
                                                  uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                  uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t y = col;
    uint32_t length = h;
    uint32_t upper = min(chunkSize*(row+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*row + 1, chunkSize*row + 1 - chunkSizeOffset);
    copyDataVertically(v, weights, m, s, wPotts, col, w, h, nc, colorOffset);
    doPottsStep(v, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, y, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}



__host__ __device__ void applyDiagonalUpperPottsSolver(float* w_, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                                       uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                       uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t length = min(h, w - col);
    uint32_t upper = min(chunkSize*(row+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*row + 1, chunkSize*row + 1 - chunkSizeOffset);
    copyDataDiagonallyUpper(w_, weights, m, s, wPotts, col, w, h, nc, colorOffset, nPotts, length);
    doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, col, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}



__host__ __device__ void applyDiagonalLowerPottsSolver(float* w_, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                                       uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                       uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t length = min(h - (col - w), w);
    uint32_t upper = min(chunkSize*(row+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*row + 1, chunkSize*row + 1 - chunkSizeOffset);
    copyDataDiagonallyLower(w_, weights, m, s, wPotts, col-w, w, h, nc, colorOffset, nPotts, length);
    doPottsStep(w_, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, col-1, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}



__host__ __device__ void applyAntiDiagonalUpperPottsSolver(float* z, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                                      uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                      uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t length = min(h, w - col);
    uint32_t upper = min(chunkSize*(row+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*row + 1, chunkSize*row + 1 - chunkSizeOffset);
    copyDataAntiDiagonallyUpper(z, weights, m, s, wPotts, col, w, h, nc, colorOffset, nPotts, length);
    doPottsStep(z, arrJ, arrP, m, s, wPotts, gamma, 0, col, w, h, nc, col, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}



__host__ __device__ void applyAntiDiagonalLowerPottsSolver(float* z, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts, float gamma,
                                                           uint32_t w, uint32_t h, uint32_t nc, uint32_t nPotts, uint32_t colorOffset, uint32_t chunkSize,
                                                           uint32_t chunkSizeOffset, uint32_t row, uint32_t col) {
    uint32_t length = min(h - (col - w), w);
    uint32_t upper = min(chunkSize*(row+1) - chunkSizeOffset, length);
    uint32_t lower = min(chunkSize*row + 1, chunkSize*row + 1 - chunkSizeOffset);
    copyDataAntiDiagonallyLower(z, weights, m, s, wPotts, col-w, w, h, nc, colorOffset, nPotts, length);
    doPottsStep(z, arrJ, arrP, m, s, wPotts, gamma, col-w, 0, w, h, nc, col-1, nPotts, length, colorOffset, upper, lower, chunkSize, chunkSizeOffset);
}


#endif