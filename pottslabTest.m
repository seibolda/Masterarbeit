close all;

addpath(genpath('Pottslab0.42'));

fig1 = figure(1);
set(fig1, 'Name', 'result', 'OuterPosition', [0, 600, 550, 500]);

imgOrg = single(double(imread('data/images/archutah.png'))) / 255;
gamma = 0.1;

%nBins = 255;
%testData = zeros(1,nBins);

%for i=0:254
%    testData(i+1) = i/nBins;
%end
%figure;
%h = histogram(testData*nBins, nBins, 'BinWidth',0.8);
%

%result = minL2Potts2DADMM(imgOrg, gamma, 'verbose', true, 'isotropic', false, 'multiThreading', true);
result = cuda_pottslab(imgOrg, gamma, 'verbose', true, 'chunkSize', 10);

imshow(result);
%figure;
%histogram(result, nBins, 'BinWidth',0.8);