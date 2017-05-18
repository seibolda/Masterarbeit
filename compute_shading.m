function [shading_energy, shading_grad_u, shading_grad_c, shading] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l, alpha, delta)

[m,n] = size(silhouette);

grad_sil = grad * u_k_plus_1(:);

p1 = reshape(grad_sil(1:m*n), m, n);
p2 = reshape(grad_sil(m*n+1:end), m, n);
grad_norm = sqrt(p1.^2 + p2.^2 + 1);
k = c_k_plus_1 ./ grad_norm;

l_dot_p = (l(1)*p1 + l(2)*p2 - l(3));

shading = l_dot_p./grad_norm;
brackets = (img - (l_dot_p./grad_norm) .* c_k_plus_1);
brackets = brackets.*silhouette;

shading_grad_c = -alpha * delta * (brackets.*(l_dot_p./ grad_norm));

shading_energy = brackets;
[~,~,dim] = size(brackets);
if dim > 1
    brackets = brackets(:,:,1) + brackets(:,:,2) + brackets(:,:,3);
    k = k(:,:,1) + k(:,:,2) + k(:,:,3);
end

shading_grad_u = alpha * delta * (-div_x*(brackets(:) .* (-l(1) .* k(:))) - div_y*(brackets(:) .* (-l(2) .* k(:))));

end