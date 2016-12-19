function grad = build_grad(m,n)

% create a row-vector containing n many ones.
e = ones(n, 1);
% Put the values of e along the lower diagonal (identified in MATLAB as -1) and
% Put the values of -e along the actual diagonal (0)
% Dx_tilde is of dimension nxn
Dx_tilde = spdiags([e -e], -1:0, n, n); 
% Take care of the Boundary conditions.
Dx_tilde(:, end) = 0;
% compute the x-derivative f_x (Output f_x is a matrix)
%sil_x = silhouette*Dx_tilde;

% Put the values of -e along the actual diagonal (0) and
% Put the values of e along the upper diagonal (1)
% Dx_tilde is of dimension nxn
e = ones(m, 1);
Dy_tilde = spdiags([-e e], 0:1, m, m);
% boundary conditions
Dy_tilde(end, end) = 0;
% compute the y-derivative f_y (Output f_y is a matrix)
%sil_y = Dy_tilde*silhouette;

Dx = kron(Dx_tilde', speye(m));
Dy = kron(speye(n), Dy_tilde);

grad = [Dx; Dy];

% imshow(silhouette);
% figure;
% imshow(reshape((Dx*silhouette(:)),m,n));
% figure;
% imshow(reshape((Dy*silhouette(:)),m,n));