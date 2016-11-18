function [f,gradf] = objfungrad(x, grad, sil, lambda, Vol)

%grad([sil==0;sil==0],:) = 0;

gx = grad * x;
N = size(x,1);
gx(1:N) = gx(1:N) .* sil;
gx(N+1:end) = gx(N+1:end) .* sil;
sq_norm_gx = gx(1:N).^2 + gx(N+1:end).^2;
norm_tmp = sqrt(1 + sq_norm_gx);
f = sum(norm_tmp) + lambda * (sum(sil.*x) - Vol)^2;
% Gradient of the objective function:
if nargout  > 1
    gradf = grad' * ([sil .* (gx(1:N) ./ norm_tmp); sil .* (gx(N+1:end) ./ norm_tmp)]) + 2*lambda*(sum(sil.*x) - Vol);
    gradf = sil .* gradf;
end