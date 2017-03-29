function u_k_plus_1 = solve_min_surface(grad, sil, lambda, Vol, u_tilde_k_plus_1, tau_u, smoothing_type)

x0 = zeros(size(sil,1),1);            % Starting guess 
%options = optimoptions(@fminunc,'Algorithm','quasi-newton');
%options = optimoptions(options,'SpecifyObjectiveGradient',true);

options = [];
options.display = 'none';
options.maxFunEvals = 200;

mygrad = @(x)objfungrad(x, grad, sil,lambda, Vol, -u_tilde_k_plus_1, tau_u, smoothing_type);
u_k_plus_1 = -minFunc(mygrad, x0, options);

%derivativeCheck(mygrad, x0);

%[u_k_plus_1,~] = fminunc(mygrad,x0,options);

