function result = cuda_pottslab(input, gamma, varargin)

% nvcc -c -Xcompiler -fPIC -std=c++11 Pottslab_cuda/src/Solver.cu
%feval('mex -largeArrayDims Pottslab_cuda/src/main_mex.cpp')
%mex -largeArrayDims Pottslab_cuda/src/main_mex.cpp Pottslab_cuda/src/Solver.o Pottslab_cuda/src/util/helper.o Pottslab_cuda/src/util/CudaBuffer.o -I"/usr/local/cuda-7.5/targets/x86_64-linux/include/" -I"/usr/local/include/opencv2" -L"/usr/local/cuda/lib64/" -lcudart -lcublas

% parse options
ip = inputParser;
addParamValue(ip, 'muStep', 2);
addParamValue(ip, 'stopTol', 1e-10);
addParamValue(ip, 'isotropic', true);
addParamValue(ip, 'verbose', false);
addParamValue(ip, 'isGPU', true);
addParamValue(ip, 'chunkSize', 0);
addParamValue(ip, 'chunkOffsetChangeType', 0);
addParamValue(ip, 'quadraticError', true);
addParamValue(ip, 'maxIterations', 50);
addParamValue(ip, 'xblocksize', 256);
addParamValue(ip, 'yblocksize', 4);

parse(ip, varargin{:});
par = ip.Results;

% check args
assert(par.muStep > 1, 'Variable muStep must be > 1.');
assert(par.stopTol > 0, 'Stopping tolerance must be > 0.');

result = main_mex(input, gamma, par.muStep, par.chunkSize, par.stopTol, par.chunkOffsetChangeType, par.maxIterations, par.verbose, par.quadraticError, par.isotropic, par.isGPU, par.xblocksize, par.yblocksize);