function u_k_plus_1 = solve_min_surface(grad, sil, lambda, Vol, u_tilde_k_plus_1, tau_u)

x0 = zeros(size(sil,1),1);            % Starting guess 
options = optimoptions(@fminunc,'Algorithm','quasi-newton');
options = optimoptions(options,'SpecifyObjectiveGradient',true);

mygrad = @(x)objfungrad(x, grad, sil,lambda, Vol, u_tilde_k_plus_1, tau_u);

[u_k_plus_1,~] = fminunc(mygrad,x0,options);