close all;

addpath(genpath('Pottslab0.42'));

%fig1 = figure(1);
%set(fig1, 'Name', 'result', 'OuterPosition', [0, 600, 550, 500]);

imgOrg = single(double(imread('data/images/coffee2_sm.png'))) / 255;
gamma = 0.1;


%f = @() minL2Potts2DADMM(imgOrg, gamma, 'verbose', false, 'isotropic', true, 'multiThreading', true);
f = @() cuda_potts_solver(imgOrg, gamma, 'verbose', false, 'isGPU', true, 'deviceNumber', 0);

%result = minL2Potts2DADMM(imgOrg, gamma, 'verbose', false, 'isotropic', true, 'multiThreading', false);
%result = cuda_pottslab(imgOrg, gamma, 'verbose', false, 'isGPU', false);

timeNeeded = timeit(f);

fprintf('Time: %f\n', timeNeeded*1000);

%imshow(result);