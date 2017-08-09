#include "PottsSolver.h"

PottsSolver::PottsSolver(float *inputImage, float newGamma, float newMuStep, size_t newW, size_t newH, size_t newNc,
                         uint32_t newChunkSize, float newStopTol, uint8_t newChunkOffsetChangeType,
                         uint32_t newMaxIterations, bool isVerbose, bool isQuadraticError) {
    h = newH;
    w = newW;
    nc = newNc;

    gamma = newGamma;
    gammaPrime = 0;
    gammaPrimeC = 0;
    gammaPrimeD = 0;
    mu = gamma * 1e-2;
    muStep = newMuStep;
    error = std::numeric_limits<float>::infinity();
    stopTol = newStopTol;
    fNorm = computeFNorm(inputImage);

    smallerDimension = std::min(h, w);
    largerDimension = std::max(h, w);
    dimension = (smallerDimension+1)*(w+h-1);

    chunkSize = largerDimension;
    if(newChunkSize > 1 && newChunkSize < largerDimension) {
        chunkSize = newChunkSize;
    }
    chunkSizeOffset = 0;

    chunkOffsetChangeType = newChunkOffsetChangeType;

    maxIterations = newMaxIterations;

    verbose = isVerbose;
    quadraticError = isQuadraticError;
}

float PottsSolver::computeFNorm(float *inputImage) {
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

void PottsSolver::updateChunkSizeOffset() {
    if(1 == chunkOffsetChangeType && chunkSize < largerDimension) {
        chunkSize+=2;
    } else if (2 == chunkOffsetChangeType && chunkSize < largerDimension) {
        chunkSizeOffset = (rand() % (chunkSize-1)) + 2;
        chunkSizeOffset = chunkSizeOffset % chunkSize;
    }
}