
close all;
clear;

addpath(genpath('minFunc_2012'));
addpath(genpath('Pottslab0.42'));
setPLJavaPath(true);

feval('mex -largeArrayDims main_mex.cpp GPUPottsSolverImpl.o helper.o CudaBuffer.o -I"/usr/local/cuda-7.5/targets/x86_64-linux/include/" -L"/usr/local/cuda/lib64/" -lcudart -lcublas')

% load data

title = 'pear';
%imgOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/light01.png')), [400, 400]))/65535;
%silhouetteOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/mask.png')), [400, 400])) > 1;
reflectanceOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/reflectance.png')), [400, 400])) / 65535;
shading_MITOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/shading.png')), [400, 400])) / 65535;

imgOrg = double(imread('data/images/archutah_med.png')) / 255;
silhouetteOrg = double(rgb2gray(imread('data/silhouettes/archutah_sil_med.png'))) > 1;

% surface
beta = 10;
lambda = 10;
Vol = 50000;
tau_u = 10;
% for pottslab
tau_c = 1;
gamma = 0.1;
% rest
alpha = 1;
delta = 1;

verbose = 1;

%for beta = 0.01:0.33:10
%    for lambda = 0.01:0.33:10
%        for gamma = 0.01:0.33:10
%            for alpha = 0.01:0.33:10
%               for delta = 0.01:033:10

                    result = singleViewReconstructionWithShapeFromShading(imgOrg, silhouetteOrg, reflectanceOrg, shading_MITOrg, lambda, beta, Vol, tau_u, tau_c, gamma, alpha, delta, verbose);

                    result = imresize(result.cdata, [600 800]);
                    imwrite(result, strcat('data/results/','u_k_',...
                        'lambda_',num2str(lambda),...
                        '_gamma_',num2str(gamma),...
                        '_alpha_',num2str(alpha),...
                        '_beta_',num2str(beta),...
                        '_delta_',num2str(delta),...
                        '.png'), 'png');

                    close all;

%               end
%            end
%        end
%    end
%end



