#ifndef _GPU_POTTS_SOLVER_H_
#define _GPU_POTTS_SOLVER_H_

#include <cmath>
#include "CudaBuffer.h"
#include "Image.h"
#include "CudaKernels.cu"

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
    CudaBuffer<float> d_error;

    dim3 block;
    dim3 grid;
    dim3 blockHorizontal;
    dim3 gridHorizontal;
    dim3 blockVertical;
    dim3 gridVertical;

    float computeFNorm(float* inputImage);

public:
    GPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc) {
        h = newH;
        w = newW;
        nc = newNc;

        gamma = newGamma;
        gammaPrime = 0;
        mu = 0.5;
        muStep = newMuStep;
        error = std::numeric_limits<float>::infinity();
        stopTol = 0.0001;
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
        d_error.CreateBuffer(1);
        d_error.SetBytewiseValue(0);

        block = dim3(32, 32, 1); // 32*32 = 1024 threads
        // ensure enough blocks to cover w * h elements (round up)
        grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, nc);
        blockHorizontal = dim3(1, 1024, 1);
        gridHorizontal = dim3(1, (h + blockHorizontal.y - 1) / blockHorizontal.y, 1);
        blockVertical = dim3(1024, 1, 1);
        gridVertical = dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);
    }

    ~GPUPottsSolver() {
        u.DestroyBuffer();
        v.DestroyBuffer();
        lam.DestroyBuffer();
        temp.DestroyBuffer();
        weights.DestroyBuffer();
        weightsPrime.DestroyBuffer();
        d_error.DestroyBuffer();
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
            prepareVerticalPottsProblems <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                    weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;


            applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(), gammaPrime, w, h, nc);
            CUDA_CHECK;
            applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(), gammaPrime, w, h, nc);
            CUDA_CHECK;


            updateLagrangeMultiplierKernel <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), lam.GetDevicePtr(),
                    temp.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;
            updateErrorKernel <<<grid, block>>> (d_error.GetDevicePtr(), temp.GetDevicePtr(), w, h, nc);
            CUDA_CHECK;
            error = d_error.DownloadData()[0];
            d_error.SetBytewiseValue(0);
            printf("Iteration: %d error: %f\n", iteration, error);
            iteration++;

            mu = mu * muStep;

            if(iteration > 50)
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

#endif