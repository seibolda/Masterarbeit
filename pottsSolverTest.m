close all;

addpath(genpath('Pottslab0.42'));
setPLJavaPath(true);
installPottslab;

imgOrg = single(double(imread('data/images/coffee2_sm.png'))) / 255;
gamma = 0.1;

[~, hostname] = system('hostname');
filename = strcat(hostname, '_resultsMatlab.txt');

fileID = fopen(filename, 'w');

for isotropic = 0:1:1
    for gamma = 0.01:0.333:3
        
        fprintf(fileID, '\n\n\n\n\n---Gamma: %f Isotropic: %d---\n\n\n', gamma, isotropic);
        
        pottslab = @() minL2Potts2DADMM(imgOrg, gamma, 'verbose', false, 'isotropic', isotropic, 'multiThreading', true);
        timePottslab = timeit(pottslab);
        fprintf(fileID, 'pottslab Time: %f\n', timePottslab*1000);

        for chunkSize = [1000 300 100 50 20 2]
            
            ownCodeCPU = @() cuda_potts_solver(imgOrg, gamma, 'verbose', false, 'isotropic', isotropic, 'isGPU', false, 'chunkOffsetChangeType', 2, 'chunkSize', chunkSize);
            timeOwnCPU = timeit(ownCodeCPU);
            fprintf(fileID, '\n--chunkSize: %d--\n\nC++ Time: %f\n', chunkSize, timeOwnCPU*1000);
            
            for xBlockSize = [1 2 4 8 16 32 64 128 256 512 1024]
                for yBlockSize = [1 2 4 8 16 32 64 128 256 512 1024]
                    if xBlockSize*yBlockSize == 1024
                        ownCodeCuda = @() cuda_potts_solver(imgOrg, gamma, 'verbose', false, 'isotropic', isotropic, 'isGPU', true, 'chunkOffsetChangeType', 2, 'chunkSize', chunkSize, 'xblocksize', xBlockSize, 'yblocksize', yBlockSize);
                        timeOwnCuda = timeit(ownCodeCuda);
                        fprintf(fileID, 'Cuda Time: %f with xBlock %d yBlock %d\n', timeOwnCuda*1000, xBlockSize, yBlockSize);
                    end
                end
            end
        end
    end
end









%fig1 = figure(1);
%set(fig1, 'Name', 'result', 'OuterPosition', [0, 600, 550, 500]);
%result = minL2Potts2DADMM(imgOrg, gamma, 'verbose', false, 'isotropic', true, 'multiThreading', false);
%result = cuda_pottslab(imgOrg, gamma, 'verbose', false, 'isGPU', false);
%imshow(result);