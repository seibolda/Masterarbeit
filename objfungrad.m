function [f, gradf] = objfungrad(x, grad, sil, lambda, Vol, u_tilde_k_plus_1, tau_u)

gx = grad * x;
N = size(x,1);
%sq_norm_gx = gx(1:N).^2 + gx(N+1:end).^2;
%sq_norm_gx = (div_x * gx(1:N)).^2 + (div_y * gx(N+1:end)).^2);
norm_laplacian = ((grad'*grad) * x).^2;

%norm_tmp = sqrt(1 + sq_norm_gx);

norm_tmp_tau = sqrt(sum((x - u_tilde_k_plus_1).^2));

%f = sum(norm_tmp) + lambda * (sum(sil.*x) - Vol)^2 + (1/(2*tau_u))*norm_tmp_tau^2;
f = 0.5 * sum(norm_laplacian) + lambda * (sum(sil.*x) - Vol)^2 + (1/(2*tau_u))*norm_tmp_tau^2;

% Gradient of the objective function:
if nargout  > 1
    gradf = (grad'*(grad*(grad'*(grad*x)))) ...grad' * ([(gx(1:N) ./ norm_tmp); (gx(N+1:end) ./ norm_tmp)]) ...
        + 2 * lambda * (sum(sil.*x) - Vol) ...
        + (x - u_tilde_k_plus_1)/tau_u;
    gradf = sil .* gradf;
end