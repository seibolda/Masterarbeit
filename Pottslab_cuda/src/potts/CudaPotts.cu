#ifndef _CUDA_POTTS_H_
#define _CUDA_POTTS_H_

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __global__
#endif


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

    while(r >  lower) {
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

#endif