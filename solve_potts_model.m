function c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma)

gamma = gamma * tau_c;
c_k_plus_1 = cuda_potts_solver(single(c_tilde_k_plus_1), gamma);
%c_k_plus_1 = minL2Potts2DADMM(c_tilde_k_plus_1, gamma, 'verbose', true);