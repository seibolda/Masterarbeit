function [saveImg] = singleViewReconstructionWithShapeFromShading(imgOrg, silhouetteOrg, reflectanceOrg, shading_MITOrg, ...
        lambda, beta, Vol, tau_u, tau_c, gamma, alpha, delta, verbose)
    
    %images
    scale = 1;
    img = imresize(imgOrg, scale);
    silhouette = imresize(silhouetteOrg, scale);
    reflectance = imresize(reflectanceOrg, scale);
    shading_MIT = imresize(shading_MITOrg, scale);

    %figures
    fig1 = figure(1);
    set(fig1, 'Name', 'u_k_plus_1 initial', 'OuterPosition', [0, 600, 550, 500]);
    fig2 = figure(2);
    set(fig2, 'Name', 'c_k_plus_1 initial', 'OuterPosition', [800, 600, 100, 100]);
    fig3 = figure(3);
    set(fig3, 'Name', 'u_k_plus_1 shading_optimized', 'OuterPosition', [0, 100, 550, 500]);
    fig4 = figure(4);
    set(fig4, 'Name', 'c_k_plus_1 shading_optimized', 'OuterPosition', [800, 100, 100, 100]);
    fig5 = figure(5);
    set(fig5, 'Name', 'shading optimized', 'OuterPosition', [1200, 100, 100, 100]);
    fig6 = figure(6);
    set(fig6, 'Name', 'error: image - shading*albedo', 'OuterPosition', [1200, 600, 100, 100]);


    %%% parameters %%%

    % for surface
    [m, n] = size(silhouette);
    Vol = Vol * scale;
    h = 1/m;
    grad = build_grad(m, n);
    %grad = grad ./ h;
    grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
    div_x = -transpose(grad(1:m*n,:));
    div_y = -transpose(grad(m*n+1:end,:));
    % for rest
    l = [1,0,-1]; % lighting vector
    l = l./(sqrt(sum(l.^2)));
    eta = 0.5;
    smoothing_type = 'gradient';
    scaling_factor = 2;

    u_tilde_k_plus_1 = zeros(m*n,1); % set constant as initialization
    %c_tilde_k_plus_1 = silhouette;
    c_tilde_k_plus_1 = img; % initialize with image
    %c_tilde_k_plus_1 = reflectance; % initialize with perfect albedo

    for resizing_step = 1:1
        % surface
        u_k_plus_1 = solve_min_surface(grad, silhouette(:), lambda, beta, Vol, u_tilde_k_plus_1, tau_u, smoothing_type);

        if verbose == 1
            figure(1);
            surfl(reshape(-u_k_plus_1,m,n));
            shading flat;
            axis ij
            axis equal
            colormap gray;
            view(-16,39);
            camlight(0,-90)
            camlight headlight
        end;


        % PottsLab

        %c_k_plus_1 = c_tilde_k_plus_1;
        c_k_plus_1 = solve_potts_model(c_tilde_k_plus_1, tau_c, gamma);

        if verbose == 1
            figure(2);
            imshow(c_k_plus_1, []);
        end;

        % estimate lighting
        img_sil = rgb2gray(img);
        img_sil = img_sil(silhouette);
        albedo = rgb2gray(c_k_plus_1);
        albedo = albedo(silhouette);
        normals = grad * u_k_plus_1;
        ux = normals(1:n*m);
        ux = ux(silhouette);
        uy = normals(n*m+1:end);
        uy = uy(silhouette);
        A = [ux, uy, -ones(size(ux,1),1)];
        A = A ./ sqrt(sum(A.^2, 2));
        A = albedo .* A;

        l = A \ img_sil;
        l = l./(sqrt(sum(l.^2)));



        for iteration_k = 0:10^(3-resizing_step)

            [shading_energy_k, shading_grad_u_k, shading_grad_c_k, shading_k] = compute_shading(u_k_plus_1, c_k_plus_1, grad, div_x, div_y, silhouette, img, l, alpha, delta);
            u_k = u_k_plus_1;
            c_k = c_k_plus_1;

            [u_k_plus_1, c_k_plus_1, u_tilde_k_plus_1, c_tilde_k_plus_1, tau_u, tau_c] = minimizing_with_line_search(tau_u, tau_c, u_k, c_k, shading_energy_k, shading_grad_u_k, shading_grad_c_k, silhouette, lambda, beta, Vol, smoothing_type, gamma, alpha, l, grad, div_x, div_y, img, eta, iteration_k, delta);

            if verbose == 1
                figure(3)
                surfl(reshape(-u_tilde_k_plus_1,m,n));
                shading flat;
                colormap gray;
                axis ij
                axis equal
                view(-16,39);
                camlight(0,-90)
                camlight headlight

                figure(4);
                imshow(c_k_plus_1, []);

                figure(5);
                imshow(shading_k);

                figure(6);
                imshow(abs(shading_energy_k),[]);
                drawnow;
            end;

        end;


        img = imresize(imgOrg, scale * (scaling_factor^resizing_step));
        silhouette = imresize(silhouetteOrg, scale * (scaling_factor^resizing_step));
        reflectance = imresize(reflectanceOrg, scale * (scaling_factor^resizing_step));
        shading_MIT = imresize(shading_MITOrg, scale * (scaling_factor^resizing_step));
        u_tilde_k_plus_1 = imresize(reshape(u_tilde_k_plus_1,m,n),scaling_factor);
        u_tilde_k_plus_1 = u_tilde_k_plus_1(:).*scaling_factor;
        c_tilde_k_plus_1 = imresize(c_tilde_k_plus_1,scaling_factor);
        [m, n] = size(silhouette);
        grad = build_grad(m, n);
        grad([silhouette(:)==0;silhouette(:)==0],:) = 0;
        div_x = -transpose(grad(1:m*n,:));
        div_y = -transpose(grad(m*n+1:end,:));
        Vol = Vol * scaling_factor^(3);
        gamma = gamma / scaling_factor;
        h = 1/m;
        grad = grad ./ h;
    end

    figure(3)
    surfl(reshape(-u_tilde_k_plus_1,m,n));
    shading flat;
    colormap gray;
    axis ij
    axis equal
    view(-16,39);
    camlight(0,-90)
    camlight headlight
    saveImg = getframe(fig3);

end