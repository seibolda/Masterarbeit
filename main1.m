silhouette = double(rgb2gray(imread('data/silhouettes/archutah_sil_sm.png'))) > 0;
[m, n] = size(silhouette);

% create a row-vector containing n many ones.
e = ones(n, 1);
% Put the values of e along the lower diagonal (identified in MATLAB as -1) and
% Put the values of -e along the actual diagonal (0)
% Dx_tilde is of dimension nxn
Dx_tilde = spdiags([e -e], -1:0, n, n); 
% Take care of the Boundary conditions.
Dx_tilde(:, end) = 0;
% compute the x-derivative f_x (Output f_x is a matrix)
sil_x = silhouette*Dx_tilde;

% Put the values of -e along the actual diagonal (0) and
% Put the values of e along the upper diagonal (1)
% Dx_tilde is of dimension nxn
e = ones(m, 1);
Dy_tilde = spdiags([-e e], 0:1, m, m);
% boundary conditions
Dy_tilde(end, end) = 0;
% compute the y-derivative f_y (Output f_y is a matrix)
sil_y = Dy_tilde*silhouette;

Dx = kron(Dx_tilde', speye(m));
Dy = kron(speye(n), Dy_tilde);

%h = silhouette(:);

imshow(silhouette);
figure;
imshow(sil_x);
figure;
imshow(sil_y);

x0 = zeros(m*n,1);            % Starting guess 
options = optimoptions(@fminunc,'Algorithm','quasi-newton');
options = optimoptions(options,'SpecifyObjectiveGradient',true);

lambda = 10;
mygrad = @(x)objfungrad(x, [Dx;Dy],silhouette(:),lambda, 10);

[x,fval] = fminunc(mygrad,x0,options);

figure;
surf(reshape(x,m,n));
