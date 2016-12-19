function c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c)

gamma = 1 * tau_c;
c_k_plus_1 = minL2Potts2DADMM(c_tilde_k_plus_1, gamma, 'verbose', true);