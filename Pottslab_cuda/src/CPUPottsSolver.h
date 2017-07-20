#ifndef POTTSLAB_CUDA_CPUPOTTSSOLVER_H
#define POTTSLAB_CUDA_CPUPOTTSSOLVER_H

#include <cstdint>
#include <cstdlib>
#include "Image.h"
#include "potts/CudaPotts.cu"

class CPUPottsSolver {
private:
    float gamma;
    float gammaPrime;
    float gammaPrimeC;
    float gammaPrimeD;
    float mu;
    float muStep;
    float error;
    float stopTol;
    float fNorm;
    uint32_t chunkSize;
    uint32_t chunkSizeOffset;

    uint32_t h;
    uint32_t w;
    uint32_t nc;

    float* in;
    float* u;
    float* v;
    float* w_;
    float* z;
    float* lam1;
    float* lam2;
    float* lam3;
    float* lam4;
    float* lam5;
    float* lam6;
    float* temp;
    float* weights;
    float* weightsPrime;

    uint32_t* arrJ;
    float* arrP;
    float* m;
    float* s;
    float* wPotts;

    size_t dimension;

    float computeFNorm(float* inputImage);

    float updateError();

    void clearHelperMemory();

    void updateChunkSizeOffset();

    void horizontalPotts4ADMM(uint32_t nHor, uint32_t colorOffset);
    void horizontalPotts8ADMM(uint32_t nHor, uint32_t colorOffsetHorVer);

    void verticalPotts4ADMM(uint32_t nVer, uint32_t colorOffset);
    void verticalPotts8ADMM(uint32_t nVer, uint32_t colorOffsetHorVer);

    void diagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags);

    void antidiagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags);

public:
    CPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc, uint32_t newChunkSize);

    ~CPUPottsSolver();

    void solvePottsProblem4ADMM();

    void solvePottsProblem8ADMM();

    void downloadOutputImage(ImageRGB outputImage);

    void downloadOutputMatlab(float* outputImage);

};

#endif //POTTSLAB_CUDA_CPUPOTTSSOLVER_H
