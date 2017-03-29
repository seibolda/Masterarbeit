close all;
clear;

% load data
scale = 0.25;
img = double(imresize(imread('data/MIT-intrinsic/cup1/light01.png'), scale))/65535;
silhouette = double(imresize(imread('data/MIT-intrinsic/cup1/mask.png'), scale)) > 1;
reflectance = double(imresize(imread('data/MIT-intrinsic/cup1/reflectance.png'), scale)) / 65535;
shading_MIT = double(imresize(imread('data/MIT-intrinsic/cup1/shading.png'), scale)) / 65535;

addpath(genpath('minFunc_2012'));
addpath(genpath('Pottslab0.42'));
setPLJavaPath(true);

%figures
fig1 = figure(1);
set(fig1, 'Name', 'u_k_plus_1 initial', 'OuterPosition', [0, 600, 550, 500]);
fig2 = figure(2);
set(fig2, 'Name', 'c_k_plus_1 initial', 'OuterPosition', [800, 600, 100, 100]);
fig3 = figure(3);
set(fig3, 'Name', 'u_k_plus_1 shading_optimized', 'OuterPosition', [0, 100, 550, 500]);
fig4 = figure(4);
set(fig4, 'Name', 'c_k_plus_1 shading_optimized', 'OuterPosition', [800, 100, 100, 100]);
fig5 = figure(5);
set(fig5, 'Name', 'shading optimized', 'OuterPosition', [1200, 100, 100, 100]);
fig6 = figure(6);
set(fig6, 'Name', 'error: image - shading*albedo', 'OuterPosition', [1200, 600, 100, 100]);


%%% parameters %%%

% for surface
[m, n] = size(silhouette);
minimal_surface_weight = 10;
lambda = 1/minimal_surface_weight;
Vol = 20000;
tau_u = 1;
grad = build_grad(m, n);
grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
div_x = -transpose(grad(1:m*n,:)); % Divergence x
div_y = -transpose(grad(m*n+1:end,:)); % Divergence y
% for pottslab
tau_c = 0.1;
gamma = 0.01/minimal_surface_weight;
% for rest
alpha = 1/minimal_surface_weight;
l = [-1,0,-1]; % lighting vector
l = l./(sqrt(sum(l.^2)));
eta = 0.5;
smoothing_type = 'gradient';





% surface

u_tilde_k_plus_1 = zeros(m*n,1); % set constant for now
u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u, smoothing_type);


figure(1);
surfl(reshape(u_k_plus_1,m,n));
shading flat;
axis ij
axis equal
colormap gray;
view(145,30);
camlight(0,-90)
camlight headlight


% PottsLab

%c_tilde_k_plus_1 = silhouette;
%c_tilde_k_plus_1 = img; % initialize with image
c_tilde_k_plus_1 = reflectance; % initialize with perfect albedo

c_k_plus_1 = c_tilde_k_plus_1;
%c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);


figure(2);
imshow(c_k_plus_1, []);





for iteration_k = 0:1000
    
    [shading_energy_k, shading_grad_u_k, shading_grad_c_k, shading_k] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l, alpha);
    u_k = u_k_plus_1;
    c_k = c_k_plus_1;
    
    %Line-search for tau_u and tau_c
    while true
        
        % Shading
        c_tilde_k_plus_1 = c_k - shading_grad_c_k*tau_c;
        u_tilde_k_plus_1 = u_k - shading_grad_u_k.*tau_u; 

        
        % Min Surface
        u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u, smoothing_type);

        % Potts
        c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);
        
        % shading
        [computed_shading, ~, ~, ~] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l, alpha);
    
        % check if step size tau_x is good
        [Q_L, p_L] = compute_correct_step_size(computed_shading, shading_energy_k, shading_grad_u_k, shading_grad_c_k, ...
            u_k_plus_1, u_k, c_k_plus_1, c_k, tau_u, tau_c, alpha);
        
        fprintf('Interation: %d p_L: %d Q_L: %d tau_u: %d tau_c: %d\n', iteration_k, p_L, Q_L, tau_u, tau_c);
        
        
        if p_L <= Q_L
            break
        end
        
        tau_u = eta * tau_u; 
        tau_c = eta * tau_c;
        
    end % end while
    
    
    figure(3)
    surfl(reshape(u_tilde_k_plus_1,m,n));
    shading flat;
    colormap gray;
    axis ij
    axis equal
    view(145,30);
	camlight(0,-90)
	camlight headlight
    
    figure(4);
    imshow(c_k_plus_1, []);
    
    figure(5);
    imshow(shading_k);
    
    figure(6);
    imshow(abs(shading_energy_k),[]);
    drawnow;
    
    
end; % end iterations


