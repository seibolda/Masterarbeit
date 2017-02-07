close all;
clear;

% load data
silhouette = double(rgb2gray(imread('data/silhouettes/wateringcan_sil_med.png'))) > 1;
img = double(rgb2gray(imread('data/images/wateringcan_med.png')))/255;

addpath(genpath('minFunc_2012'));
addpath(genpath('Pottslab0.42'));
setPLJavaPath(true);

%figures
fig1 = figure(1);
set(fig1, 'Name', 'u_k_plus_1 initial', 'OuterPosition', [0, 500, 550, 500]);
fig2 = figure(2);
set(fig2, 'Name', 'c_k_plus_1 initial', 'OuterPosition', [800, 500, 100, 100]);
fig3 = figure(3);
set(fig3, 'Name', 'u_k_plus_1_shading optimized', 'OuterPosition', [0, 0, 550, 500]);
fig4 = figure(4);
set(fig4, 'Name', 'c_k_plus_1_shading optimized', 'OuterPosition', [800, 0, 100, 100]);


%%% parameters %%%

% for surface
[m, n] = size(silhouette);
lambda = 0.01;
Vol = 100000;
tau_u = 10;
grad = build_grad(m, n);
grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
div_x = -transpose(grad(1:m*n,:)); % Divergence x
div_y = -transpose(grad(m*n+1:end,:)); % Divergence y
% for pottslab
tau_c = 1;
gamma = 25;
% for rest
alpha = 1;
l = [1,0,1]; % lighting vector
eta = 0.5;





% surface

u_tilde_k_plus_1 = zeros(m*n,1); % set constant for now
u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u);


figure(1);
surfl(reshape(u_k_plus_1,m,n));
shading flat;
axis equal
colormap gray;
view(145,30);
camlight(0,-90)
camlight headlight


% PottsLab

%c_tilde_k_plus_1 = silhouette;
c_tilde_k_plus_1 = img; % initialize with image

%c_k_plus_1 = c_tilde_k_plus_1;
c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);


figure(2);
imshow(c_k_plus_1);





for iteration = 0:100
    
    [computed_shading_k, shading_grad_u, shading_grad_c] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l);
    u_k = u_k_plus_1;
    c_k = c_k_plus_1;
    
    %Line-search for tau_u and tau_c
    while true
        
        % Shading
        c_tilde_k_plus_1 = c_k + alpha*(shading_grad_c*tau_c);
        u_tilde_k_plus_1 = u_k + alpha*(shading_grad_u.*tau_u); 

        
        % Min Surface
        u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u);

        % Potts
        c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);
        
        % shading
        [computed_shading, shading_grad_u, shading_grad_c] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l);
    
        % g_x = sum(sqrt(1 + grad_u(1:m*n).^2 + grad_u(m*n+1:end).^2)) + lambda * (sum(silhouette(:).*u_k_plus_1) - Vol)^2 + gamma * sum(abs(grad_c) > 1e-6);
        % check if step size tau_x is good
        [Q_L, p_L] = compute_correct_step_size(computed_shading, computed_shading_k, shading_grad_u, shading_grad_c, ...
            u_k_plus_1, u_k, c_k_plus_1, c_k, tau_u, tau_c, alpha);
        
        fprintf('p_L: %d Q_L: %d tau_u: %d tau_c: %d\n', p_L, Q_L, tau_u, tau_c);
        
        
        if p_L <= Q_L
            break
        end
        
        % update step size tau_x
        tau_u = eta * tau_u; 
        tau_c = eta * tau_c;
        
    end % end while
    
    
    figure(3)
    surfl(reshape(u_tilde_k_plus_1,m,n));
    shading flat;
    colormap gray;
    axis equal
    view(145,30);
	camlight(0,-90)
	camlight headlight
    
    figure(4);
    imshow(reshape(c_tilde_k_plus_1,m,n));
    drawnow;
    
    
end; % end iterations


