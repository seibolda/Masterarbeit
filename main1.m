silhouette = double(rgb2gray(imread('data/silhouettes/archutah_sil_sm.png'))) > 0;
img = double(imread('data/images/archutah.png'))/255;
[m, n] = size(silhouette);

lambda = 10;
Vol = 10;
grad = build_grad(m, n);
grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
u_tilde_k_plus_1 = ones(m*n,1); % set constant for now
tao_u = 0.01;

u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tao_u);

figure;
surf(reshape(u_k_plus_1,m,n));

% PottsLab
gamma = 1;
%u = minL2Potts2DADMM(img, gamma, 'verbose', true);
%figure;
%imshow(u);