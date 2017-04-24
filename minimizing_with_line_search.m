function [u_k_plus_1, c_k_plus_1,  u_tilde_k_plus_1, c_tilde_k_plus_1, tau_u, tau_c] = minimizing_with_line_search(tau_u, tau_c, u_k,...
    c_k, shading_energy_k, shading_grad_u_k, shading_grad_c_k, silhouette, lambda, Vol, smoothing_type, gamma, alpha, l, grad,...
    div_x, div_y, img, eta, iteration_k)

while true

    % Shading
    c_tilde_k_plus_1 = c_k - shading_grad_c_k*tau_c;
    u_tilde_k_plus_1 = u_k - shading_grad_u_k.*tau_u; 

    % Min Surface
    u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, Vol, u_tilde_k_plus_1, tau_u, smoothing_type);

    % Potts
    c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);

    % shading
    [computed_shading, ~, ~, ~] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l, alpha);

    % check if step size tau_x is good
    [Q_L, p_L] = compute_correct_step_size(computed_shading, shading_energy_k, shading_grad_u_k, shading_grad_c_k, ...
        u_k_plus_1, u_k, c_k_plus_1, c_k, tau_u, tau_c, alpha);

    fprintf('Iteration: %d p_L: %d Q_L: %d tau_u: %d tau_c: %d\n', iteration_k, p_L, Q_L, tau_u, tau_c);

    if p_L <= Q_L
        break
    end

    tau_u = eta * tau_u; 
    tau_c = eta * tau_c;

end
end