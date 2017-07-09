#ifndef _CUDA_COPY_DATA_H_
#define _CUDA_COPY_DATA_H_

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#endif

#include <cstdint>
#include <cmath>

__host__ __device__ float normQuad(float* array, uint32_t position, uint32_t nc, uint32_t colorOffset) {
    float result = 0;
    for(uint32_t c = 0; c < nc; c++) {
        result += array[position + c*colorOffset]*array[position + c*colorOffset];
    }
    return result;
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




#endif