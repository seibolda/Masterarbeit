#ifndef _GPU_POTTS_SOLVER_H_
#define _GPU_POTTS_SOLVER_H_

#include <cmath>
#include <cstdlib>
#include "util/CudaBuffer.h"
//#include "Image.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "PottsSolver.h"

class GPUPottsSolver : public PottsSolver{
private:

    CudaBuffer<float> d_inputImage;
    CudaBuffer<float> u;
    CudaBuffer<float> v;
    CudaBuffer<float> w_;
    CudaBuffer<float> z;
    CudaBuffer<float> lam1;
    CudaBuffer<float> lam2;
    CudaBuffer<float> lam3;
    CudaBuffer<float> lam4;
    CudaBuffer<float> lam5;
    CudaBuffer<float> lam6;
    CudaBuffer<float> temp;
    CudaBuffer<float> weights;
    CudaBuffer<float> weightsPrime;

    CudaBuffer<uint32_t> arrJ;
    CudaBuffer<float> arrP;
    CudaBuffer<float> m;
    CudaBuffer<float> s;
    CudaBuffer<float> wPotts;

    dim3 block;
    dim3 grid;
    dim3 blockHorizontal;
    dim3 gridHorizontal;
    dim3 blockVertical;
    dim3 gridVertical;
    dim3 blockDiagonal;
    dim3 gridDiagonal;
    dim3 gridSwap;

    cublasHandle_t cublasHandle;

    float updateError();

    void clearHelperMemory();

    void horizontalPotts4ADMM(uint32_t nHor, uint32_t colorOffset);
    void horizontalPotts8ADMM(uint32_t nHor, uint32_t colorOffsetHorVer);

    void verticalPotts4ADMM(uint32_t nVer, uint32_t colorOffset);
    void verticalPotts8ADMM(uint32_t nVer, uint32_t colorOffsetHorVer);

    void diagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags);

    void antidiagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags);

public:
    GPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc,
                   uint32_t newChunkSize, float newStopTol, uint8_t newChunkOffsetChangeType,
                   uint32_t newMaxIterations, bool isVerbose, bool isQuadraticError, uint32_t newXBlockSize, uint32_t newYBlockSize);

    ~GPUPottsSolver();

    void solvePottsProblem4ADMM();

    void solvePottsProblem8ADMM();

//    void downloadOutputImage(ImageRGB outputImage);

    float* getResultPtr();

    void downloadOutputMatlab(float* outputImage);

};

#endif