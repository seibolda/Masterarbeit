#ifndef _GPU_POTTS_SOLVER_H_
#define _GPU_POTTS_SOLVER_H_

#include "CudaBuffer.h"
#include "Image.h"
#include "CudaKernels.cu"

class GPUPottsSolver {
private:
    float gamma;
    size_t h;
    size_t w;
    size_t nc;

    CudaBuffer<float> d_inputImage;
    CudaBuffer<float> d_outputImage;

    dim3 block;
    dim3 grid;

public:
    GPUPottsSolver(float* inputImage, float gamma, size_t newW, size_t newH, size_t newNc) {
        h = newH;
        w = newW;
        nc = newNc;

        d_inputImage.CreateBuffer(h*w*nc);
        d_inputImage.UploadData(inputImage);

        d_outputImage.CreateBuffer(h*w*nc);
        d_outputImage.SetBytewiseValue(0);

        block = dim3(32, 32, 1); // 32*32 = 1024 threads
        // ensure enough blocks to cover w * h elements (round up)
        grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, nc);
    }

    ~GPUPottsSolver() {
        d_inputImage.DestroyBuffer();
        d_outputImage.DestroyBuffer();
    }

    void copyTest() {
        copyTestKernel <<<grid, block>>> (d_inputImage.GetDevicePtr(), d_outputImage.GetDevicePtr(), w, h, nc);
    }

    void downloadOuputImage(ImageRGB outputImage) {
        outputImage.SetRawData(d_outputImage.DownloadData());
    }

};

#endif