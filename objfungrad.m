function [f,gradf] = objfungrad(x, grad, sil, lambda, Vol)

gx = grad * x;
N = size(x,1);
sq_norm_gx = gx(1:N).^2 + gx(N+1:end).^2;
norm_tmp = sqrt(1 + sq_norm_gx);

f = sum(norm_tmp) + lambda * (sum(sil.*x) - Vol)^2;

% Gradient of the objective function:
if nargout  > 1
    gradf = grad' * ([(gx(1:N) ./ norm_tmp); (gx(N+1:end) ./ norm_tmp)]) + 2*lambda*(sum(sil.*x) - Vol);
    gradf = sil .* gradf;
end