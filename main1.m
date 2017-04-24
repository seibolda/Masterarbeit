close all;
clear;

addpath(genpath('minFunc_2012'));
addpath(genpath('Pottslab0.42'));
setPLJavaPath(true);

% load data

title = 'pear';
%imgOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/light01.png')), [400, 400]))/65535;
%silhouetteOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/mask.png')), [400, 400])) > 1;
reflectanceOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/reflectance.png')), [400, 400])) / 65535;
shading_MITOrg = double(imresize(imread(strcat('data/MIT-intrinsic/',title,'/shading.png')), [400, 400])) / 65535;

imgOrg = double(imread('data/images/archutah_med.png')) / 255;
silhouetteOrg = double(rgb2gray(imread('data/silhouettes/archutah_sil_med.png'))) > 1;

% surface
minimal_surface_weight = 10;
lambda = 1/minimal_surface_weight;
Vol = 100000;
tau_u = 10;
% for pottslab
tau_c = 1;
gamma = 1/minimal_surface_weight;
% rest
alpha = 1/minimal_surface_weight;

verbose = 0;

for minimal_surface_weight = 0.01:0.33:10
    for lambda = 0.01:0.33:10
        for gamma = 0.01:0.33:10
            for alpha = 0.01:0.33:10

                result = singleViewReconstructionWithShapeFromShading(imgOrg, silhouetteOrg, reflectanceOrg, shading_MITOrg, minimal_surface_weight, lambda, Vol, tau_u, tau_c, gamma, alpha, verbose);

                result = imresize(result.cdata, [600 800]);
                imwrite(result, strcat('data/results/','u_k_',...
                    'lambda_',num2str(lambda),...
                    '_gamma_',num2str(gamma),...
                    '_alpha_',num2str(alpha),...
                    '_minSurfWeight_',num2str(minimal_surface_weight),...
                    '.png'), 'png');

                close all;

            end
        end
    end
end



