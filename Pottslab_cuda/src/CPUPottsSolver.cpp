#include "CPUPottsSolver.h"

CPUPottsSolver::CPUPottsSolver(float *inputImage, float newGamma, float newMuStep, size_t newW, size_t newH,
                               size_t newNc, uint32_t newChunkSize) {
    h = newH;
    w = newW;
    nc = newNc;

    gamma = newGamma;
    gammaPrime = 0;
    mu = gamma * 1e-1;
    muStep = newMuStep;
    error = std::numeric_limits<float>::infinity();
    stopTol = 1e-2;
    fNorm = computeFNorm(inputImage);
    chunkSize = newChunkSize;

    u = malloc(h*w*nc*sizeof(float));
    v = malloc(h*w*nc*sizeof(float));
    w_ = malloc(h*w*nc*sizeof(float));
    z = malloc(h*w*nc*sizeof(float));
    lam1 = malloc(h*w*nc*sizeof(float));
    lam2 = malloc(h*w*nc*sizeof(float));
    lam3 = malloc(h*w*nc*sizeof(float));
    lam4 = malloc(h*w*nc*sizeof(float));
    lam5 = malloc(h*w*nc*sizeof(float));
    lam6 = malloc(h*w*nc*sizeof(float));
    temp = malloc(h*w*nc*sizeof(float));
    weights = malloc(h*w*sizeof(float));
    weightsPrime = malloc(h*w*sizeof(float));

    uint32_t smallerDimension = min(h, w);
    size_t dimension = (smallerDimension+1)*(w+h-1);
    arrJ = malloc((dimension*2+1)* sizeof(uint32_t));
    arrP = malloc(dimension* sizeof(float));
    m = malloc(dimension* sizeof(float));
    s = malloc(dimension* sizeof(float));
    wPotts = malloc(dimension* sizeof(float));
}

CPUPottsSolver::~CPUPottsSolver() {
    delete u;
    delete v;
    delete w_;
    delete z;
    delete lam1;
    delete lam2;
    delete lam3;
    delete lam4;
    delete lam5;
    delete lam6;
    delete temp;
    delete weights;
    delete weightsPrime;

    delete arrJ;
    delete arrP;
    delete m;
    delete s;
    delete wPotts;
}

float CPUPottsSolver::computeFNorm(float* inputImage) {
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