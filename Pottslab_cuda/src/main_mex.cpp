#include "mex.h"

#include "GPUPottsSolver.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // input validation
    if (nrhs != 2 || nlhs > 1)
        mexErrMsgTxt("Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("Inputs must be single arrays.");
    if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Inputs must be real arrays.");
    if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
        mexErrMsgTxt("Inputs must be dense arrays.");

    // create ouput array
    mwSize numel = mxGetNumberOfElements(prhs[0]);
    mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *dims = mxGetDimensions(prhs[0]);
    plhs[0] = mxCreateNumericArray(ndims, dims, mxSINGLE_CLASS, mxREAL);

    mexPrintf("%d %d %d %d\n", ndims, dims[0], dims[1], dims[2]);

    // get pointers to data
    float* output = (float *) mxGetData(plhs[0]);
    float* input = (float *) mxGetData(prhs[0]);
    float gamma = (float) mxGetScalar(prhs[1]);

    GPUPottsSolver gpuPottsSolver(input, gamma, 2, dims[0], dims[1], dims[2]);

    gpuPottsSolver.solvePottsProblem4ADMM();

    gpuPottsSolver.downloadOutputMatlab(output);
}