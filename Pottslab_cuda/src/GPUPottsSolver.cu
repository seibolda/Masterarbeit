#ifndef _GPU_POTTS_SOLVER_IMPL_H_
#define _GPU_POTTS_SOLVER_IMPL_H_

#include "GPUPottsSolver.h"
#include "CudaKernels.cu"
#include "CudaKernels4ADMM.cu"
#include "CudaKernels8ADMM.cu"

GPUPottsSolver::GPUPottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc) {
    h = newH;
    w = newW;
    nc = newNc;

    gamma = newGamma;
    gammaPrime = 0;
    mu = gamma * 1e-2;
    muStep = newMuStep;
    error = std::numeric_limits<float>::infinity();
    stopTol = 1e-5;
    fNorm = computeFNorm(inputImage);

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

    uint32_t smallerDimension = min(h, w);
    size_t dimension = (smallerDimension+1)*(w+h-1);
    arrJ.CreateBuffer(dimension);
    arrJ.SetBytewiseValue(0);
    arrP.CreateBuffer(dimension*2+1);
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
    gridHorizontal = dim3((h + blockHorizontal.x - 1) / blockHorizontal.x, 1, 1);
    blockVertical = dim3(1024, 1, 1);
    gridVertical = dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);//dim3((w + blockVertical.x - 1) / blockVertical.x, 1, 1);
    blockDiagonal = dim3(1024, 1, 1);
    gridDiagonal = dim3((h + w + blockDiagonal.x - 1) / blockDiagonal.x, 1, 1);

//    uint32_t largerDimension = max(h, w);
//    gridSwap = dim3((largerDimension + block.x - 1), (largerDimension + block.y - 1), nc);

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

void GPUPottsSolver::clearHelperMemory() {
    arrJ.SetBytewiseValue(0);
    arrP.SetBytewiseValue(0);
    m.SetBytewiseValue(0);
    s.SetBytewiseValue(0);
    wPotts.SetBytewiseValue(0);
}

float GPUPottsSolver::updateError() {
    float errorCublas = 0;
//    CUBLAS_CHECK(cublasSnrm2(cublasHandle, h*w*nc, temp.GetDevicePtr(), 1, &errorCublas));
    CUBLAS_CHECK(cublasSasum(cublasHandle, h*w*nc, temp.GetDevicePtr(), 1, &errorCublas));
    return errorCublas;
}

void GPUPottsSolver::solvePottsProblem4ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffset = (w+1)*(h+1);

    ImageRGB testImage(w, h);

    setWeightsKernel <<<grid, block>>> (weights.GetDevicePtr(), w, h);

    gammaPrime = 2 * gamma;

    while (error >= stopTol * fNorm) {

        updateWeightsPrimeKernel <<<grid, block>>> (weightsPrime.GetDevicePtr(), weights.GetDevicePtr(), w, h, mu, 1);
        CUDA_CHECK;


        prepareHorizontalPottsProblems4ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam1.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;
        applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime,
                w, h, nc, nHor, colorOffset);
        CUDA_CHECK;
        copyHorizontallyTest <<<grid, block>>> (u.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc);
        CUDA_CHECK;
        clearHelperMemory();

//        testImage.SetRawData(u.DownloadData());
//        testImage.Show("Test Image", 100+w, 100);
//        cv::waitKey(0);


        prepareVerticalPottsProblems4ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                weights.GetDevicePtr(), weightsPrime.GetDevicePtr(), lam1.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;
        applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrime,
                w, h, nc, nVer, colorOffset);
        CUDA_CHECK;
        copyVerticallyTest <<<grid, block>>> (v.GetDevicePtr(), arrJ.GetDevicePtr(), m.GetDevicePtr(), wPotts.GetDevicePtr(), w, h, nc);
        CUDA_CHECK;
        clearHelperMemory();




        updateLagrangeMultiplierKernel4ADMM <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), lam1.GetDevicePtr(),
                temp.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;



        error = updateError();
        printf("Iteration: %d error: %f\n", iteration, error);
        iteration++;

        mu = mu * muStep;

        if(iteration > 30)
            break;
    }

}

void GPUPottsSolver::solvePottsProblem8ADMM() {
    if (0 == fNorm) {
        return;
    }
    uint32_t iteration = 0;

    uint32_t nHor = w;
    uint32_t nVer = h;
    uint32_t colorOffsetHorVer = (w+1)*(h+1);

    uint32_t nDiags = min(h, w);
    uint32_t colorOffsetDiags = (min(h, w)+1)*(w+h-1);

    ImageRGB testImage(w, h);

    float omegaC = sqrt(2.0) - 1.0;
    float omegaD = 1.0 - sqrt(2.0)/2.0;
    gammaPrimeC = 4.0 * omegaC * gamma;
    gammaPrimeD = 4.0 * omegaD * gamma;

    setWeightsKernel <<<grid, block>>> (weights.GetDevicePtr(), w, h);

    while (error >= stopTol * fNorm) {
        updateWeightsPrimeKernel <<<grid, block>>> (weightsPrime.GetDevicePtr(), weights.GetDevicePtr(), w, h, mu, 6);
        CUDA_CHECK;

        // Horizontal
        prepareHorizontalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                lam1.GetDevicePtr(), lam2.GetDevicePtr(), lam3.GetDevicePtr(), mu, w, h ,nc);
        CUDA_CHECK;
        applyHorizontalPottsSolverKernel<<<gridHorizontal, blockHorizontal>>> (u.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeC,
                w, h, nc, nHor, colorOffsetHorVer);
        CUDA_CHECK;
        clearHelperMemory();


        // Diagonal
        prepareDiagonalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                lam2.GetDevicePtr(), lam4.GetDevicePtr(), lam6.GetDevicePtr(), mu, w, h ,nc);
        CUDA_CHECK;
        applyDiagonalPottsSolverKernel<<<gridDiagonal, blockDiagonal>>> (w_.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeD,
                w, h, nc, nDiags,colorOffsetDiags);
        CUDA_CHECK;
        clearHelperMemory();



        // Vertical
        prepareVerticalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                lam1.GetDevicePtr(), lam4.GetDevicePtr(), lam5.GetDevicePtr(), mu, w, h ,nc);
        CUDA_CHECK;
        applyVerticalPottsSolverKernel<<<gridVertical, blockVertical>>> (v.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeC,
                w, h, nc, nVer, colorOffsetHorVer);
        CUDA_CHECK;
        clearHelperMemory();



        // Antidiagonal
        prepareAntidiagonalPottsProblems8ADMM <<<grid, block>>> (d_inputImage.GetDevicePtr(), u.GetDevicePtr(), v.GetDevicePtr(),
                w_.GetDevicePtr(), z.GetDevicePtr(), weights.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                lam3.GetDevicePtr(), lam5.GetDevicePtr(), lam6.GetDevicePtr(), mu, w, h ,nc);
        CUDA_CHECK;
        applyAntiDiagonalPottsSolverKernel<<<gridDiagonal, blockDiagonal>>> (z.GetDevicePtr(), weightsPrime.GetDevicePtr(),
                arrJ.GetDevicePtr(), arrP.GetDevicePtr(), m.GetDevicePtr(), s.GetDevicePtr(), wPotts.GetDevicePtr(), gammaPrimeD,
                w, h, nc, nDiags, colorOffsetDiags);
        CUDA_CHECK;
        clearHelperMemory();

//        testImage.SetRawData(w_.DownloadData());
//        testImage.Show("Test Image", 100+w, 100);
//        cv::waitKey(0);



        updateLagrangeMultiplierKernel8ADMM <<<grid,block>>> (u.GetDevicePtr(), v.GetDevicePtr(), w_.GetDevicePtr(), z.GetDevicePtr(),
                lam1.GetDevicePtr(), lam2.GetDevicePtr(), lam3.GetDevicePtr(),
                lam4.GetDevicePtr(), lam5.GetDevicePtr(), lam6.GetDevicePtr(),
                temp.GetDevicePtr(), mu, w, h, nc);
        CUDA_CHECK;

        error = updateError();
        printf("Iteration: %d error: %f\n", iteration, error);
        iteration++;

        mu = mu * muStep;

        if(iteration > 30)
            break;
    }
}

void GPUPottsSolver::downloadOuputImage(ImageRGB outputImage) {
    outputImage.SetRawData(u.DownloadData());
}

void GPUPottsSolver::downloadOutputMatlab(float *outputImage) {
    cudaMemcpy(outputImage, u.GetDevicePtr(), h*w*nc*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
}

/*void GPUPottsSolver::swapTest() {
    ImageRGB rotatedImage(h, w);
    ImageRGB image(w,h);

    swapImageCCWKernel <<<grid, block>>> (v.GetDevicePtr(), tempV.GetDevicePtr(), w, h, nc);
    CUDA_CHECK;
    swapImageCWKernel <<<grid, block>>> (tempV.GetDevicePtr(), v.GetDevicePtr(), h, w, nc);
    CUDA_CHECK;

    image.SetRawData(v.DownloadData());
    image.Show("image", 100, 100);
    rotatedImage.SetRawData(tempV.DownloadData());
    rotatedImage.Show("Rotated Image", 100+w, 100);
    cv::waitKey(0);
}*/

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