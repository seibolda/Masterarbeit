close all;
clear;

silhouette = double(rgb2gray(imread('data/silhouettes/wateringcan_sil_med.png'))) > 1;
img = double(rgb2gray(imread('data/images/wateringcan_med.png')))/255;

addpath(genpath('minFunc_2012'));
addpath(genpath('Pottslab0.42'));
%setPLJavaPath(true);

%figures
fig1 = figure(1);
set(fig1, 'Name', 'u_k_plus_1 initial', 'OuterPosition', [0, 500, 550, 500]);
fig2 = figure(2);
set(fig2, 'Name', 'c_k_plus_1 initial', 'OuterPosition', [800, 500, 100, 100]);
fig3 = figure(3);
set(fig3, 'Name', 'u_k_plus_1_shading optimized', 'OuterPosition', [0, 0, 550, 500]);
fig4 = figure(4);
set(fig4, 'Name', 'c_k_plus_1_shading optimized', 'OuterPosition', [800, 0, 100, 100]);


% surface
[m, n] = size(silhouette);
lambda = 0.01;
Vol = 100000;
grad = build_grad(m, n);
grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
u_tilde_k_plus_1 = zeros(m*n,1); % set constant for now
tau_u = 1;

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
tau_c = 0.1;
gamma = 15;

%c_tilde_k_plus_1 = silhouette;
c_tilde_k_plus_1 = img; % initialize with image

%c_k_plus_1 = c_tilde_k_plus_1;
c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);


figure(2);
imshow(c_k_plus_1);




% lighting vector
l = [1,0,1];

% Divergence
div_x = -transpose(grad(1:m*n,:));
div_y = -transpose(grad(m*n+1:end,:));

alpha = 1;

for i = 0:100
	% Shading
    grad_sil = grad * u_k_plus_1(:);

	p1 = reshape(grad_sil(1:m*n),m,n);
	p2 = reshape(grad_sil(m*n+1:end),m,n);
	grad_norm = p1.^2+p2.^2+1;
	k = c_k_plus_1 ./ grad_norm;
	
	l_dot_p = (-l(1)*p1-l(2)*p2+l(3));
	
	brackets = (img-l_dot_p./grad_norm.*c_k_plus_1);
	brackets = brackets.*silhouette;
	
	c_tilde_k_plus_1 = c_k_plus_1  + alpha*(tau_c*brackets.*l_dot_p./ grad_norm);
	
	u_k_plus_1_shading = u_k_plus_1 + alpha*((div_x*(brackets(:) .* (l(1) .* k(:))) + div_y*(brackets(:) .* (l(2) .* k(:)))).*tau_u); 
	
    
    
    
	figure(3)
    surfl(reshape(u_k_plus_1_shading,m,n));
    shading flat;
    colormap gray;
    axis equal
    view(145,30);
	camlight(0,-90)
	camlight headlight
		
	% Min Surface
    u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_k_plus_1_shading, tau_u);
    
    % Potts
    c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);
    
    
    
    
    figure(4);
    imshow(reshape(c_tilde_k_plus_1,m,n));
    drawnow;
    
    
end;


