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
        weights[index] = 1;
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

        targetImg[index] = (weights[weightsIndex] * inputImage[index] + sourceImg[index] * mu - lam[index]) / weightsPrime[weightsIndex];

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

__device__ float normQuadKernel(float* array, uint32_t position, uint32_t colorOffset) {
    return (array[position]*array[position]
            + array[position + colorOffset]*array[position + colorOffset]
            + array[position + 2*colorOffset]*array[position + 2*colorOffset]);
}

__device__ void doPottsStepKernel(float* arrayToUpdate, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* w,
                                  float gamma, uint32_t rowCol, uint32_t n, uint32_t widthOrHeight, uint32_t nc) {
    uint32_t y = rowCol;
    uint32_t h = widthOrHeight;

//    uint32_t* arrJ = new uint32_t[n];
//    float* arrP = new float[n];
    float d = 0;
    float p = 0;
    float dpg = 0;
//    float* m = new float[(n + 1) * nc];
//    float* s = new float[n + 1];
//    float* w = new float[n + 1];
//    float* mu = new float[nc];
    float mu1, mu2, mu3;



    m[0] = 0;
    m[n+1] = 0;
    m[2*(n+1)] = 0;
    s[0] = 0;
    w[0] = 0;
    float wTemp, mTemp, wDiffTemp;
    for(uint32_t j = 0; j < n; j++) {
        wTemp = weights[j + y*n];
        m[j + 1 + y*(n+1)] = arrayToUpdate[j + y*n] * wTemp + m[j + y*(n+1)];
        m[j + 1 + y*(n+1) + (n+1)*h] = arrayToUpdate[j + y*n + n*h] * wTemp + m[j + y*(n+1) + (n+1)*h];
        m[j + 1 + y*(n+1) + 2*(n+1)*h] = arrayToUpdate[j + y*n + 2*n*h] * wTemp + m[j + y*(n+1) + 2*(n+1)*h];
        s[j + 1 + y*(n+1)] = normQuadKernel(arrayToUpdate, j+y*n, n*h) * wTemp + s[j + y*(n+1)];
        w[j + 1 + y*(n+1)] = w[j + y*(n+1)] + wTemp;
    }

    for(uint32_t r = 1; r <= n; r++) {
        arrP[r - 1 + y*n] = s[r + y*(n+1)] - normQuadKernel(m, r + y*(n+1), (n+1)*h) / w[r + y*(n+1)];
        arrJ[r - 1 + y*n] = 0;
        for(uint32_t l = r; l >= 2; l--) {
            mTemp = 0;
            for(uint32_t k = 0; k < nc; k++) {
                mTemp = mTemp + (m[r + y*(n+1) + k*(n+1)*h]*m[l - 1 + y*(n+1) + k*(n+1)*h]) * (m[r + y*(n+1) + k*(n+1)*h]*m[l - 1 + y*(n+1) + k*(n+1)*h]);
            }
            wDiffTemp = w[r + y*(n+1)] - w[l - 1 + y*(n+1)];
            if(0 == wDiffTemp) {
                d = 0;
            } else {
                d = s[r + y*(n+1)] - s[l - 1 + y*(n+1)] - mTemp / wDiffTemp;
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
        mu1 = (m[r + y*(n+1)] - m[l + y*(n+1)]) / (w[r + y*(n+1)] - w[l + y*(n+1)]);
        mu2 = (m[r + y*(n+1) + (n+1)*h] - m[l + y*(n+1) + (n+1)*h]) / (w[r + y*(n+1)] - w[l + y*(n+1)]);
        mu3 = (m[r + y*(n+1) + 2*(n+1)*h] - m[l + y*(n+1) + 2*(n+1)*h]) / (w[r + y*(n+1)] - w[l + y*(n+1)]);

        for(uint32_t j = l; j < r; j++) {
            arrayToUpdate[j + y*n] = mu1;
            arrayToUpdate[j + y*n + n*h] = mu2;
            arrayToUpdate[j + y*n + 2*n*h] = mu3;
        }
        r = l;
        if (r < 1) break;
        l = arrJ[r - 1 + y*n];
    }


//    delete[] arrJ;
//    delete[] arrP;
//    delete[] m;
//    delete[] s;
//    delete[] w;
//    delete[] mu;
}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                                 float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(y < h) {
        doPottsStepKernel(u, weights, arrJ, arrP, m, s, wPotts, gamma, y, w, h, nc);
    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, uint32_t* arrJ, float* arrP, float* m, float* s, float* wPotts,
                                               float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if(x < w) {
        doPottsStepKernel(v, weights, arrJ, arrP, m, s, wPotts, gamma, x, h, w, nc);
    }
}


#endif