#ifndef _GPU_POTTS_SOLVER_H_
#define _GPU_POTTS_SOLVER_H_

#include <cmath>
#include "CudaBuffer.h"
#include "Image.h"
#include "CudaKernels.cu"
#include "cublas_v2.h"

class GPUPottsSolver {
private:
    float gamma;
    float gammaPrime;
    float mu;
    float muStep;
    float error;
    float stopTol;
    float fNorm;

    uint32_t h;
    uint32_t w;
    uint32_t nc;

    CudaBuffer<float> d_inputImage;
    CudaBuffer<float> u;
    CudaBuffer<float> v;
    CudaBuffer<float> lam;
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

    cublasHandle_t cublasHandle;

    float computeFNorm(float* inputImage);

    float updateError();

public:
    GPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc) {
        h = newH;
        w = newW;
        nc = newNc;

        gamma = newGamma;
        gammaPrime = 0;
        mu = gamma * 1e-2;
        muStep = newMuStep;
        error = std::numeric_limits<float>::infinity();
        stopTol = 1e-10;
        fNorm = computeFNorm(inputImage);

        d_inputImage.CreateBuffer(h*w*nc);
        d_inputImage.UploadData(inputImage);
        u.CreateBuffer(h*w*nc);
        u.SetBytewiseValue(0);
        v.CreateBuffer(h*w*nc);
        v.UploadData(inputImage);
        lam.CreateBuffer(h*w*nc);
        lam.SetBytewiseValue(0);
        temp.CreateBuffer(h*w*nc);
        temp.SetBytewiseValue(0);
        weights.CreateBuffer(w*h);
        weights.SetBytewiseValue(0);
        weightsPrime.CreateBuffer(w*h);
        weightsPrime.SetBytewiseValue(0);

        arrJ.CreateBuffer(h*w);
        arrJ.SetBytewiseValue(0);
        arrP.CreateBuffer(h*w);
        arrP.SetBytewiseValue(0);
        m.CreateBuffer((h+1)*(w+1)*nc);
        m.SetBytewiseValue(0);
        s.CreateBuffer((h+1)*(w+1));
        s.SetBytewiseValue(0);
        wPotts.CreateBuffer((h+1)*(w+1));
        wPotts.SetBytewiseValue(0);

        block = dim3(32, 32, 1); // 32*32 = 1024 threads
        // ensure enough blocks to cover w * h elements (round up)
        grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, nc);
        blockHorizontal = dim3(1, 1024, 1);
        gridHorizontal = dim3(1, (h + blockHorizontal.y - 1) / blockHorizontal.y, 1);
        blockVertical = dim3(1024, 1, 1);
        gridVertical = dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);

        CUBLAS_CHECK(cublasCreate(&cublasHandle));
    }

    ~GPUPottsSolver() {
        u.DestroyBuffer();
        v.DestroyBuffer();
        lam.DestroyBuffer();
        temp.DestroyBuffer();
        weights.DestroyBuffer();
        weightsPrime.DestroyBuffer();

        arrP.DestroyBuffer();
        arrJ.DestroyBuffer();
        m.DestroyBuffer();
        s.DestroyBuffer();
        wPotts.DestroyBuffer();

        CUBLAS_CHECK(cublasDestroy(cublasHandle));

    }

    void solvePottsProblem() {
        if (0 == fNorm) {
            return;
        }
        uint32_t iteration = 0;

        setWeightsKernel <<<grid, block>>> (weights.GetDevicePtr(), w, h);

        gammaPrime = 2 * gamma;

        while (error >= stopTol * fNorm) {

            updateWeightsPrimeKernel <<<grid, block>>> (weightsPrime.GetDevicePtr(), weights.GetDevicePtr(), w, h, mu);
            CUDA_CHECK;


            prepareHorizontalPottsProblems <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                    weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;
            applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                    arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime, w, h, nc);
            CUDA_CHECK;


            prepareVerticalPottsProblems <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                    weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;
            applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                    arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime, w, h, nc);
            CUDA_CHECK;


            updateLagrangeMultiplierKernel <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), lam.GetDevicePtr(),
                    temp.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;



            error = updateError();
            printf("Iteration: %d error: %f\n", iteration, error);
            iteration++;

            mu = mu * muStep;

            if(iteration > 25)
                break;
        }

    }

    void downloadOuputImage(ImageRGB outputImage) {
        outputImage.SetRawData(u.DownloadData());
    }

};

float GPUPottsSolver::computeFNorm(float* inputImage) {
    float fNorm = 0;
    for(uint32_t x = 0; x < w; x++) {
        for(uint32_t y = 0; y < h; y++) {
            for(uint32_t c = 0; c < nc; c++) {
                fNorm += pow(inputImage[x + y * w + c * w * h], 2);
            }
        }
    }
    return fNorm;
}

float GPUPottsSolver::updateError() {
    float errorCublas = 0;
    CUBLAS_CHECK(cublasSnrm2(cublasHandle, h*w*nc, temp.GetDevicePtr(), 1, &errorCublas)) ;
    return errorCublas*errorCublas;
}

#endif