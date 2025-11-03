function plot_som_umatrix(som, outpath)
% PLOT_SOM_UMATRIX Visualize SOM U-Matrix (unified distance matrix)
%
% Inputs:
%   som     - SOM model structure
%   outpath - Optional output file path
%
% Theory: U-Matrix shows distances between neighboring prototypes.
%         Dark regions = large distances (cluster boundaries)
%         Light regions = small distances (within clusters)
%         Reveals topological organization from SOM's cooperation phase.

    K = som.num_neurons;
    gs = som.grid_size;
    W = som.weights;

    % Compute average distance to 4-neighbors for each neuron
    U = zeros(gs);
    W2 = reshape(W, [gs(1), gs(2), size(W, 2)]);

    for i = 1:gs(1)
        for j = 1:gs(2)
            nb = [];
            % Collect neighbors
            if i > 1, nb = [nb; squeeze(W2(i - 1, j, :))']; end
            if i < gs(1), nb = [nb; squeeze(W2(i + 1, j, :))']; end
            if j > 1, nb = [nb; squeeze(W2(i, j - 1, :))']; end
            if j < gs(2), nb = [nb; squeeze(W2(i, j + 1, :))']; end

            % Current neuron weight
            w = squeeze(W2(i, j, :))';

            % Compute Euclidean distance to each neighbor
            d = sqrt(sum((nb - w).^2, 2));
            U(i, j) = mean(d);
        end
    end

    % Visualization
    figure('Color', 'w', 'Position', [100, 100, 700, 600]);
    set(gcf, 'ToolBar', 'none', 'MenuBar', 'none');
    imagesc(U);
    axis image off;
    colormap(parula);
    cb = colorbar;
    set(cb, 'Color', 'k');
    title('SOM U-Matrix (Unified Distance)', 'FontSize', 12, 'FontWeight', 'bold');

    if nargin > 1
        saveas(gcf, outpath);
        close(gcf);
    end
end
