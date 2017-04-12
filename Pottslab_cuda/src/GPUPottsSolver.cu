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

    dim3 block;
    dim3 grid;

    float computeFNorm(float* inputImage);

public:
    GPUPottsSolver(float* inputImage, float newGamma, size_t newW, size_t newH, size_t newNc) {
        h = newH;
        w = newW;
        nc = newNc;

        gamma = newGamma;
        gammaPrime = 0;
        mu = 1;
        error = std::numeric_limits<float>::infinity();
        stopTol = 1;
        fNorm = computeFNorm(inputImage);

        d_inputImage.CreateBuffer(h*w*nc);
        d_inputImage.UploadData(inputImage);
        u.CreateBuffer(h*w*nc);
        u.UploadData(inputImage);
        v.CreateBuffer(h*w*nc);
        v.UploadData(inputImage);
        lam.CreateBuffer(h*w*nc);
        lam.SetBytewiseValue(0);
        temp.CreateBuffer(h*w*nc);
        temp.SetBytewiseValue(0);
        weights.CreateBuffer(w*h);
        weights.SetBytewiseValue(1);
        weightsPrime.CreateBuffer(w*h);
        weightsPrime.SetBytewiseValue(0);

        block = dim3(32, 32, 1); // 32*32 = 1024 threads
        // ensure enough blocks to cover w * h elements (round up)
        grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, nc);
    }

    ~GPUPottsSolver() {
        u.DestroyBuffer();
        v.DestroyBuffer();
        lam.DestroyBuffer();
        temp.DestroyBuffer();
    }

    void solvePottsProblem() {
        if (fNorm == 0) {
            return;
        }
        while (error >= stopTol * fNorm) {
            //updateWeightsPrimeKernel <<<grid, block>>> (weights.GetDevicePtr());
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