function [Q_L, p_L] = compute_correct_step_size(computed_shading, computed_shading_k, shading_grad_u, shading_grad_c, ...
    u_k_plus_1, u_k, c_k_plus_1, c_k, ...
    lambda, silhouette, tau_u, tau_c, grad, Vol, alpha, gamma)

%[m,n] = size(silhouette);
%grad_u = grad * u_k_plus_1;
%grad_c = grad * c_k_plus_1(:);
g_x = 0;%sum(sqrt(1 + grad_u(1:m*n).^2 + grad_u(m*n+1:end).^2)) + lambda * (sum(silhouette(:).*u_k_plus_1) - Vol)^2 + gamma * sum(abs(grad_c) > 1e-6);

Q_L = alpha * 0.5 * sum(computed_shading_k(:).^2) ...
    + sum(((u_k_plus_1 - u_k) .* shading_grad_u) + ((c_k_plus_1(:) - c_k(:)) .* shading_grad_c(:))) ...
    + sum((1/(2*tau_u)) * (u_k_plus_1 - u_k).^2 + (1/(2*tau_c)) * (c_k_plus_1(:) - c_k(:)).^2);
p_L = alpha * 0.5 * sum(computed_shading(:).^2);

end