#ifndef _GPU_POTTS_SOLVER_H_
#define _GPU_POTTS_SOLVER_H_

#include <cmath>
#include "CudaBuffer.h"
#include "Image.h"
#include "CudaKernels.cu"
#include "cublas_v2.h"
#include <cstdlib>

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
    CudaBuffer<float> tempV;
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
    dim3 gridSwap;

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
        tempV.CreateBuffer(h*w*nc);
        tempV.SetBytewiseValue(0);
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
        gridVertical = dim3((h + blockVertical.x - 1) / blockVertical.x, 1, 1);//dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);
        uint32_t largerDimension = max(h, w);
        gridSwap = dim3((largerDimension + block.x - 1), (largerDimension + block.y - 1), nc);

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

        ImageRGB testImage(w, h);

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
            arrJ.SetBytewiseValue(0);
            arrP.SetBytewiseValue(0);
            m.SetBytewiseValue(0);
            s.SetBytewiseValue(0);
            wPotts.SetBytewiseValue(0);


            prepareVerticalPottsProblems <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                    weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;
            swapImageCWKernel <<<gridSwap, block>>> (v.GetDevicePtr(), tempV.GetDevicePtr(), w, h, nc);
            CUDA_CHECK;
            applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (tempV.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                    arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime, w, h, nc);
            CUDA_CHECK;
            swapImageCCWKernel <<<gridSwap, block>>> (tempV.GetDevicePtr(), v.GetDevicePtr(), h, w, nc);
            CUDA_CHECK;

//            testImage.SetRawData(v.DownloadData());
//            testImage.Show("Test Image", 100+w, 100);
//            cv::waitKey(0);


            updateLagrangeMultiplierKernel <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), lam.GetDevicePtr(),
                    temp.GetDevicePtr(), mu, w, h, nc);
            CUDA_CHECK;



            error = updateError();
            printf("Iteration: %d error: %f\n", iteration, error);
            iteration++;

            mu = mu * muStep;

            if(iteration > 100)
                break;
        }

    }

    void downloadOuputImage(ImageRGB outputImage) {
        outputImage.SetRawData(v.DownloadData());
    }

    void doPottsOnCPU();

    void swapTest() {
        ImageRGB rotatedImage(h, w);

        swapImageCCWKernel <<<grid, block>>> (v.GetDevicePtr(), tempV.GetDevicePtr(), w, h, nc);
        CUDA_CHECK;
        swapImageCWKernel <<<grid, block>>> (tempV.GetDevicePtr(), v.GetDevicePtr(), h, w, nc);
        CUDA_CHECK;

        rotatedImage.SetRawData(tempV.DownloadData());
        rotatedImage.Show("Rotated Image", 100+w, 100);
        cv::waitKey(0);
    }

};

float GPUPottsSolver::computeFNorm(float* inputImage) {
    float fNorm = 0;
    for(uint32_t x = 0; x < w; x++) {
        for(uint32_t y = 0; y < h; y++) {
            for(uint32_t c = 0; c < nc; c++) {
                fNorm += pow(inputImage[x + y*w + c*w*h], 2);
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

void GPUPottsSolver::doPottsOnCPU() {
    uint32_t problemSize = 255;
    uint32_t height1 = 1;
    uint32_t numCh = 1;
    float* weights1 = new float[problemSize*height1];
    float* weightsPrime1 = new float[problemSize*height1];
    float* m1 = new float[(problemSize+1)*(height1+1)*numCh];
    float* s1 = new float[(problemSize+1)*(height1+1)];
    float* w1 = new float[(problemSize+1)*(height1+1)];
    float* arrP1 = new float[problemSize*height1];
    uint32_t* arrJ1 = new uint32_t[problemSize*height1];

    float* testData = new float[255];
    int* inputToVisualize = new int[255];
    int* result = new int[255];

    float* u = new float[problemSize*height1*numCh];
    float* v = new float[problemSize*height1*numCh];
    float* lam = new float[problemSize*height1*numCh];
    float* temp = new float[problemSize*height1*numCh];

    for(uint32_t x = 0; x < problemSize; x++) {
        for(uint32_t y = 0; y < height1; y++) {

            uint32_t weightsIndex = x + problemSize*y;
            weights1[weightsIndex] = 1;
            weightsPrime1[weightsIndex] = 0;
            arrJ1[weightsIndex] = 0;
            arrP1[weightsIndex] = 0;

            for(uint32_t c = 0; c < numCh; c++) {
                uint32_t index = x + problemSize*y + c*height1*problemSize;
                u[index] = 0;
                v[index] = 0;
                lam[index] = 0;
                temp[index] = 0;
            }
        }
    }
    for(uint32_t i = 0; i < (problemSize+1)*(height1+1); i++) {

        s1[i] = 0;
        w1[i] = 0;

        for(uint32_t c = 0; c < numCh; c++) {
            m1[i*c] = 0;
        }
    }
    for(uint32_t i = 0; i < problemSize*height1*numCh; ++i) {
        testData[i] = i/255.0;//(rand() % 255) / 255.0;
        inputToVisualize[i] = floor(testData[i] * 255);

    }



    // prepare horizontal
    for(uint32_t x = 0; x < problemSize; x++) {
        for(uint32_t y = 0; y < height1; y++) {
            for(uint32_t c = 0; c < numCh; c++) {
                uint32_t weightsIndex = x + problemSize*y;
                uint32_t index = x + problemSize*y + c*problemSize*height1;
                u[index] = (testData[index] * weights1[weightsIndex] + v[index] * mu - lam[index]) / weightsPrime1[weightsIndex];
                v[index] = (testData[index] * weights1[weightsIndex] + u[index] * mu + lam[index]) / weightsPrime1[weightsIndex];
            }
        }
    }




    /*showHistogram256("testData", inputToVisualize, 10,10);

    for(uint32_t i = 0; i < (problemSize+1)*(height1+1)*numCh; i++) {
        m1[i] = 0;
    }
    w1[0] = 0;
    s1[0] = 0;

    for(uint32_t i = 0; i < 1; ++i) {
        doPottsStep(testData, weights1, arrJ1, arrP1, m1, s1, w1, gamma*2, 0, problemSize, height1, numCh);
    }

    for(uint32_t i = 0; i < 256; ++i) {
        result[i] = floor(testData[i] * 255);
        if(i >= problemSize*numCh)
            result[i] = 0;
        printf("Pos: %d data: %f res: %d\n", i, testData[i], result[i]);
    }

    showHistogram256("result", result, 10,200);*/

    delete[] weights1;
    delete[] weightsPrime1;
    delete[] m1;
    delete[] w1;
    delete[] s1;
    delete[] arrJ1;
    delete[] arrP1;
    delete[] testData;
    delete[] inputToVisualize;
    delete[] result;

    delete[] u;
    delete[] v;
    delete[] lam;
    delete[] temp;
}

#endif