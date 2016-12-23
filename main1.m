silhouette = double(rgb2gray(imread('data/silhouettes/wateringcan_sil_med.png'))) > 0;
img = double(rgb2gray(imread('data/images/wateringcan_med.png')))/255;



% surface
[m, n] = size(silhouette);
lambda = 10;
Vol = 10;
grad = build_grad(m, n);
grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
u_tilde_k_plus_1 = zeros(m*n,1); % set constant for now
tau_u = 1000;

u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u);

figure;
surf(reshape(u_k_plus_1,m,n));



% PottsLab
setPLJavaPath(true);
tau_c = 1;
gamma = 0.01;
c_tilde_k_plus_1 = img; % initialize with image

c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);
figure;
imshow(c_k_plus_1);



% do step
grad_sil = grad * silhouette(:);
l = [1,0,1];
c = img(:);
img_vec = img(:);

for x = 1:m
    for y = 0:(n-1)
        idx = x + y*m;
        grad_norm = sqrt(grad_sil(idx)^2 + grad_sil(idx + m*n)^2 + 1);
        k = c(idx) / grad_norm;
        p = [grad_sil(idx), grad_sil(idx + m*n), -1];
        brackets = (img_vec(idx) - (dot(p, l) / grad_norm));
        
        u_tilde_k_plus_1(idx) = u_k_plus_1(idx) - ([grad(idx); grad(idx + m*n)]' * [brackets * (l(1) * k); brackets * (l(2) * k)]);
        c_tilde_k_plus_1(idx) = c_k_plus_1(idx) + brackets * ((dot(p,l) / grad_norm) * c_k_plus_1(idx));
    end
end

figure;
surf(reshape(u_tilde_k_plus_1,m,n));
figure;
surf(reshape(c_tilde_k_plus_1,m,n));