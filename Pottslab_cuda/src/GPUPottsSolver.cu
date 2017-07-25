#ifndef _GPU_POTTS_SOLVER_IMPL_H_
#define _GPU_POTTS_SOLVER_IMPL_H_

#include "GPUPottsSolver.h"
#include "CudaKernels.cu"
#include "potts/CudaKernels4ADMM.cu"
#include "potts/CudaKernels8ADMM.cu"

GPUPottsSolver::GPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc,
                               uint32_t newChunkSize, float newStopTol, uint8_t newChunkOffsetChangeType,
                               uint32_t newMaxIterations, bool isVerbose, bool isQuadraticError) :
        PottsSolver(inputImage, newGamma, newMuStep, newW, newH, newNc, newChunkSize, newStopTol, newChunkOffsetChangeType,
        newMaxIterations, isVerbose, isQuadraticError) {

    d_inputImage.CreateBuffer(h*w*nc);
    d_inputImage.UploadData(inputImage);
    u.CreateBuffer(h*w*nc);
    u.SetBytewiseValue(0);
    v.CreateBuffer(h*w*nc);
    v.UploadData(inputImage);
    w_.CreateBuffer(h*w*nc);
    w_.UploadData(inputImage);
    z.CreateBuffer(h*w*nc);
    z.UploadData(inputImage);
    lam1.CreateBuffer(h*w*nc);
    lam1.SetBytewiseValue(0);
    lam2.CreateBuffer(h*w*nc);
    lam2.SetBytewiseValue(0);
    lam3.CreateBuffer(h*w*nc);
    lam3.SetBytewiseValue(0);
    lam4.CreateBuffer(h*w*nc);
    lam4.SetBytewiseValue(0);
    lam5.CreateBuffer(h*w*nc);
    lam5.SetBytewiseValue(0);
    lam6.CreateBuffer(h*w*nc);
    lam6.SetBytewiseValue(0);
    temp.CreateBuffer(h*w*nc);
    temp.SetBytewiseValue(0);
    weights.CreateBuffer(w*h);
    weights.SetBytewiseValue(0);
    weightsPrime.CreateBuffer(w*h);
    weightsPrime.SetBytewiseValue(0);

    arrJ.CreateBuffer(dimension*2+1);
    arrJ.SetBytewiseValue(0);
    arrP.CreateBuffer(dimension);
    arrP.SetBytewiseValue(0);
    m.CreateBuffer((dimension)*nc);
    m.SetBytewiseValue(0);
    s.CreateBuffer(dimension);
    s.SetBytewiseValue(0);
    wPotts.CreateBuffer(dimension);
    wPotts.SetBytewiseValue(0);

    block = dim3(32, 32, 1); // 32*32 = 1024 threads
    // ensure enough blocks to cover w * h elements (round up)
    grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, nc);
    blockHorizontal = dim3(1024, 1, 1);
    gridHorizontal = dim3((h + blockHorizontal.x - 1) / blockHorizontal.x, (ceil(((double)w / (double)chunkSize)) + 1 + blockHorizontal.y - 1) / blockHorizontal.y, 1);
    blockVertical = dim3(1024, 1, 1);
    gridVertical = dim3((w + blockVertical.x - 1) / blockVertical.x, (ceil(((double)h / (double)chunkSize)) + 1 + blockHorizontal.y - 1) / blockHorizontal.y, 1);//dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);
    blockDiagonal = dim3(1024, 1, 1);
    gridDiagonal = dim3((h + w + blockDiagonal.x - 1) / blockDiagonal.x, (ceil(((double)w / (double)chunkSize)) + 1 + blockDiagonal.y - 1) / blockDiagonal.y, 1);

    CUBLAS_CHECK(cublasCreate(&cublasHandle));
}

GPUPottsSolver::~GPUPottsSolver() {
    u.DestroyBuffer();
    v.DestroyBuffer();
    w_.DestroyBuffer();
    z.DestroyBuffer();
    lam1.DestroyBuffer();
    lam2.DestroyBuffer();
    lam3.DestroyBuffer();
    lam4.DestroyBuffer();
    lam5.DestroyBuffer();
    lam6.DestroyBuffer();
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

void GPUPottsSolver::clearHelperMemory() {
    arrJ.SetBytewiseValue(0);
    arrP.SetBytewiseValue(0);
    m.SetBytewiseValue(0);
    s.SetBytewiseValue(0);
    wPotts.SetBytewiseValue(0);
}

float GPUPottsSolver::updateError() {
    float errorCublas = 0;
    if(quadraticError) {
        CUBLAS_CHECK(cublasSnrm2(cublasHandle, h*w*nc, temp.GetDevicePtr(), 1, &errorCublas));
        errorCublas = errorCublas*errorCublas;
    } else {
        CUBLAS_CHECK(cublasSasum(cublasHandle, h*w*nc, temp.GetDevicePtr(), 1, &errorCublas));
    }

    return errorCublas;
}

void GPUPottsSolver::horizontalPotts4ADMM(uint32_t nHor, uint32_t colorOffset) {
    prepareHorizontalPottsProblems4ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam1.GetDevicePtr(), mu, w, h, nc);
    CUDA_CHECK;
    applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime,
            w, h, nc, nHor, colorOffset, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackHorizontally <<<grid, block>>> (u.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc, colorOffset);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::verticalPotts4ADMM(uint32_t nVer, uint32_t colorOffset) {
    prepareVerticalPottsProblems4ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam1.GetDevicePtr(), mu, w, h, nc);
    CUDA_CHECK;
    applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime,
            w, h, nc, nVer, colorOffset, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackVertically <<<grid, block>>> (v.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc, colorOffset);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::solvePottsProblem4ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffset = (w+1)*(h+1);

    float stopThreshold = stopTol * fNorm;

//    ImageRGB testImage(w, h);

    setWeightsKernel <<<grid, block>>> (weights.GetDevicePtr(), w, h);

    gammaPrime = 2 * gamma;

    while (error >= stopThreshold) {

        updateWeightsPrimeKernel <<<grid, block>>> (weightsPrime.GetDevicePtr(), weights.GetDevicePtr(), w, h, mu, 1);
        CUDA_CHECK;

        horizontalPotts4ADMM(nHor, colorOffset);

        verticalPotts4ADMM(nVer, colorOffset);

//        testImage.SetRawData(u.DownloadData());
//        testImage.Show("Test Image", 100+w, 100);
//        cv::waitKey(0);

        updateLagrangeMultiplierKernel4ADMM <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), lam1.GetDevicePtr(),
                temp.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;



        error = updateError();

        if (verbose) {
            printf("Iteration: %d error: %f\n", iteration, error);
        }

        iteration++;

        mu = mu * muStep;

        updateChunkSizeOffset();

        if(iteration > maxIterations)
            break;
    }

}

void GPUPottsSolver::horizontalPotts8ADMM(uint32_t nHor, uint32_t colorOffsetHorVer) {
    prepareHorizontalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            lam1.GetDevicePtr(), lam2.GetDevicePtr(), lam3.GetDevicePtr(), mu, w, h ,nc);
    CUDA_CHECK;
    applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeC,
            w, h, nc, nHor, colorOffsetHorVer, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackHorizontally <<<grid, block>>> (u.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc, colorOffsetHorVer);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::verticalPotts8ADMM(uint32_t nVer, uint32_t colorOffsetHorVer) {
    prepareVerticalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            lam1.GetDevicePtr(), lam4.GetDevicePtr(), lam5.GetDevicePtr(), mu, w, h ,nc);
    CUDA_CHECK;
    applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeC,
            w, h, nc, nVer, colorOffsetHorVer, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackVertically <<<grid, block>>> (v.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc, colorOffsetHorVer);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::diagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags) {
    prepareDiagonalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            lam2.GetDevicePtr(), lam4.GetDevicePtr(), lam6.GetDevicePtr(), mu, w, h ,nc);
    CUDA_CHECK;
    applyDiagonalPottsSolverKernel<<<gridDiagonal, blockDiagonal>>> (w_.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeD,
            w, h, nc, nDiags,colorOffsetDiags, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackDiagonallyUpper <<<grid, block>>> (w_.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(),
            w, h, nc, colorOffsetDiags, nDiags);
    CUDA_CHECK;
    copyBackDiagonallyLower <<<grid, block>>> (w_.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(),
            w, h, nc, colorOffsetDiags, nDiags);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::antidiagonalPotts8ADMM(uint32_t nDiags, uint32_t colorOffsetDiags) {
    prepareAntidiagonalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
            w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            lam3.GetDevicePtr(), lam5.GetDevicePtr(), lam6.GetDevicePtr(), mu, w, h ,nc);
    CUDA_CHECK;
    applyAntiDiagonalPottsSolverKernel<<<gridDiagonal, blockDiagonal>>> (z.GetDevicePtr(), weightsPrime.GetDevicePtr(),
            arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeD,
            w, h, nc, nDiags, colorOffsetDiags, chunkSize, chunkSizeOffset);
    CUDA_CHECK;
    copyBackAntiDiagonallyUpper <<<grid, block>>> (z.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(),
            w, h, nc, colorOffsetDiags, nDiags);
    CUDA_CHECK;
    copyBackAntiDiagonallyLower <<<grid, block>>> (z.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(),
            w, h, nc, colorOffsetDiags, nDiags);
    CUDA_CHECK;
    clearHelperMemory();
}

void GPUPottsSolver::solvePottsProblem8ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    float stopThreshold = stopTol * fNorm;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffsetHorVer = (w+1)*(h+1);

    uint32_t nDiags = min(h, w);
    uint32_t colorOffsetDiags = (min(h, w)+1)*(w+h-1);

//    ImageRGB testImage(w, h);

    float omegaC = sqrt(2.0) - 1.0;
    float omegaD = 1.0 - sqrt(2.0)/2.0;
    gammaPrimeC = 4.0 * omegaC * gamma;
    gammaPrimeD = 4.0 * omegaD * gamma;

    setWeightsKernel <<<grid, block>>> (weights.GetDevicePtr(), w, h);

    while (error >= stopThreshold) {
        updateWeightsPrimeKernel <<<grid, block>>> (weightsPrime.GetDevicePtr(), weights.GetDevicePtr(), w, h, mu, 6);
        CUDA_CHECK;

        horizontalPotts8ADMM(nHor, colorOffsetHorVer);

        diagonalPotts8ADMM(nDiags, colorOffsetDiags);

        verticalPotts8ADMM(nVer, colorOffsetHorVer);

        antidiagonalPotts8ADMM(nDiags, colorOffsetDiags);

//        testImage.SetRawData(u.DownloadData());
//        testImage.Show("Test Image", 100+w, 100);
//        cv::waitKey(0);

        updateLagrangeMultiplierKernel8ADMM <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), w_.GetDevicePtr(), z.GetDevicePtr(),
                lam1.GetDevicePtr(), lam2.GetDevicePtr(), lam3.GetDevicePtr(),
                lam4.GetDevicePtr(), lam5.GetDevicePtr(), lam6.GetDevicePtr(),
                temp.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;

        error = updateError();

        if (verbose) {
            printf("Iteration: %d error: %f\n", iteration, error);
        }

        iteration++;

        mu = mu * muStep;

        updateChunkSizeOffset();

        if(iteration > maxIterations)
            break;
    }
}

//void GPUPottsSolver::downloadOutputImage(ImageRGB outputImage) {
//    outputImage.SetRawData(u.DownloadData());
//}

float* GPUPottsSolver::getResultPtr() {
    return u.DownloadData();
}

void GPUPottsSolver::downloadOutputMatlab(float *outputImage) {
    cudaMemcpy(outputImage, u.GetDevicePtr(), h*w*nc*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
}


#endif