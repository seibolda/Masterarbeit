#ifndef POTTSLAB_CUDA_POTTSSOLVER_H
#define POTTSLAB_CUDA_POTTSSOLVER_H

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <algorithm>

class PottsSolver {

protected:
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

    size_t dimension;
    uint32_t smallerDimension;
    uint32_t largerDimension;

    uint8_t chunkOffsetChangeType;

    bool verbose;
    bool quadraticError;

    uint32_t maxIterations;

    float computeFNorm(float* inputImage);

    void updateChunkSizeOffset();

public:
    PottsSolver(float* inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc,
                uint32_t newChunkSize, float newStopTol, uint8_t chunkOffsetChangeType, uint32_t newMaxIterations,
                bool isVerbose, bool isQuadraticError);

};


#endif //POTTSLAB_CUDA_POTTSSOLVER_H
