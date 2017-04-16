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

__device__ float normQuadKernel(float* array, uint32_t position, uint32_t colorOffset) {
    return (array[position]*array[position]
            + array[position + colorOffset]*array[position + colorOffset]
            + array[position + 2*colorOffset]*array[position + 2*colorOffset]);
}

__device__ void doPottsStepKernel(float* arrayToUpdate, float* weights, float gamma, uint32_t ) {

}

__global__ void applyHorizontalPottsSolverKernel(float* u, float* weights, float gamma, uint32_t n, uint32_t h, uint32_t nc) {
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(y < h) {
        uint32_t* arrJ = new uint32_t[n];
        float* arrP = new float[n];
        float d = 0;
        float p = 0;
        float dpg = 0;
        float* m = new float[(n + 1) * nc];
        float* s = new float[n + 1];
        float* w = new float[n + 1];
        float* mu = new float[nc];



        m[0] = 0;
        m[n+1] = 0;
        m[2*(n+1)] = 0;
        s[0] = 0;
        w[0] = 0;
        float wTemp, mTemp, wDiffTemp;
        for(uint32_t j = 0; j < n; j++) {
            wTemp = weights[j + y*n];
            m[j + 1] = u[j + y*n] * wTemp + m[j];
            m[j + 1 + n+1] = u[j + y*n + n*h] * wTemp + m[j + n+1];
            m[j + 1 + 2*(n+1)] = u[j + y*n + 2*n*h] * wTemp + m[j + 2*(n+1)];
            s[j + 1] = normQuadKernel(u, j+y*n, n*h) * wTemp + s[j];
            w[j + 1] = w[j] + wTemp;
//            if(y == 2)
//                printf("y: %d m1: %f m2: %f m3: %f s: %f w: %f\n", y, m[j+1], m[j+1+n+1], m[j+1+2*(n+1)], s[j+1], w[j+1]);
        }

        for(uint32_t r = 1; r <= n; r++) {
            arrP[r - 1] = s[r] - normQuadKernel(m, r, n+1) / w[r];
            arrJ[r - 1] = 0;
            for(uint32_t l = r; l >= 2; l--) {
                mTemp = 0;
                for(uint32_t k = 0; k < nc; k++) {
                    mTemp = mTemp + (m[r + k*(n+1)]*m[l - 1 + k*(n+1)]) * (m[r + k*(n+1)]*m[l - 1 + k*(n+1)]);
                }
                wDiffTemp = w[r] - w[l - 1];
                if(0 == wDiffTemp) {
                    d = 0;
                } else {
                    d = s[r] - s[l - 1] - mTemp / wDiffTemp;
                }
                dpg = d + gamma;
                if(dpg > arrP[r-1]) {
                    break;
                }
                p = arrP[l - 2] + dpg;
                if(p < arrP[r - 1]) {
                    arrP[r - 1] = p;
                    arrJ[r - 1] = l - 1;
                }
            }
        }

        uint32_t r = n;
        uint32_t l = arrJ[r - 1];
        while(r > 0) {
            for(uint32_t k = 0; k < nc; k++) {
                mu[k] = (m[r + k*(n+1)] - m[l + k*(n+1)]) / (w[r] - w[l]);

            }
            for(uint32_t j = l; j < r; j++) {
                for(uint32_t k = 0; k < nc; k++) {
                    u[j + y*n + k*n*h] = mu[k];
                }
            }
            r = l;
            if (r < 1) break;
            l = arrJ[r - 1];
        }


        delete[] arrJ;
        delete[] arrP;
        delete[] m;
        delete[] s;
        delete[] w;
        delete[] mu;
    }
}

__global__ void applyVerticalPottsSolverKernel(float* v, float* weights, float gamma, uint32_t w, uint32_t h, uint32_t nc) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if(x < w) {
        printf("col: %d\n", x);
    }
}


#endif