#include "mex.h"

#include "PottsSolver.h"
#include "GPUPottsSolver.h"
#include "CPUPottsSolver.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // input validation
    if (nrhs < 11 || nlhs > 1)
        mexErrMsgTxt("Wrong number of input/output arguments.");
    /*if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("Inputs must be single arrays.");*/
    if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Inputs must be real arrays.");
    if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
        mexErrMsgTxt("Inputs must be dense arrays.");

    // create ouput array
    mwSize numel = mxGetNumberOfElements(prhs[0]);
    mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *dims = mxGetDimensions(prhs[0]);
    plhs[0] = mxCreateNumericArray(ndims, dims, mxSINGLE_CLASS, mxREAL);

    // get pointers to data
    float* output = (float *) mxGetData(plhs[0]);
    float* input = (float *) mxGetData(prhs[0]);

    // set values
    float gamma = (float) mxGetScalar(prhs[1]);
    int width = dims[0];
    int height = dims[1];
    int numberChannels = dims[2];
    float muStep = (float) mxGetScalar(prhs[2]);
    int chunkSize = (int) mxGetScalar(prhs[3]);
    float stopTol = (float) mxGetScalar(prhs[4]);
    int chunkOffsetChangeType = (int) mxGetScalar(prhs[5]);
    int maxIterations = (int) mxGetScalar(prhs[6]);
    bool verbose = (bool) mxGetScalar(prhs[7]);
    bool quadraticError = (bool) mxGetScalar(prhs[8]);
    bool isotropic = (bool) mxGetScalar(prhs[9]);
    bool isGPU = (bool) mxGetScalar(prhs[10]);
    int xBlockSize = (int) mxGetScalar(prhs[11]);
    int yBlockSize = (int) mxGetScalar(prhs[12]);

//    mexPrintf("gamma: %f width: %d height: %d numCh: %d muStep: %f chunkSize: %d stopTol: %f chunkChange: %d maxIt: %d verbose: %d quadError: %d isotrop: %d gpu: %d\n",
//              gamma, width, height, numberChannels, muStep, chunkSize, stopTol, chunkOffsetChangeType, maxIterations, verbose, quadraticError, isotropic, isGPU);

    if(isGPU) {
        GPUPottsSolver gpuPottsSolver(input, gamma, muStep, width, height, numberChannels, chunkSize,
                                      stopTol, chunkOffsetChangeType, maxIterations, verbose, quadraticError, xBlockSize, yBlockSize);

        if(isotropic) {
            gpuPottsSolver.solvePottsProblem8ADMM();
        } else {
            gpuPottsSolver.solvePottsProblem4ADMM();
        }

        gpuPottsSolver.downloadOutputMatlab(output);
    } else {
        CPUPottsSolver cpuPottsSolver(input, gamma, muStep, width, height, numberChannels, chunkSize,
                                      stopTol, chunkOffsetChangeType, maxIterations, verbose, quadraticError);

        if(isotropic) {
            cpuPottsSolver.solvePottsProblem8ADMM();
        } else {
            cpuPottsSolver.solvePottsProblem4ADMM();
        }

        cpuPottsSolver.downloadOutputMatlab(output);
    }

//    GPUPottsSolver gpuPottsSolver(input, gamma, 2, dims[0], dims[1], dims[2]);
//
//    gpuPottsSolver.solvePottsProblem4ADMM();
//
//    gpuPottsSolver.downloadOutputMatlab(output);
}