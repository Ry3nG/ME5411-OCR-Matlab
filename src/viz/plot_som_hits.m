function hit_stats = plot_som_hits(patches, som, outpath)
% PLOT_SOM_HITS Visualize SOM hit map (neuron activation frequency)
%
% Inputs:
%   patches - [M x D] normalized patch matrix
%   som     - SOM model structure
%   outpath - Optional output file path
%
% Output (optional):
%   hit_stats - struct with fields:
%       .hits        : [K x 1] raw hit counts
%       .grid_hits   : [grid_size] reshaped hit map
%       .top_indices : indices of top-10 neurons
%       .top_counts  : corresponding hit counts
%
% Theory: Hit map shows how often each neuron is BMU (Best Matching Unit).
%         Reveals data distribution across the learned codebook.
%         High-hit neurons = frequently occurring visual patterns.

    W = som.weights;
    K = som.num_neurons;
    gs = som.grid_size;

    % Find BMU for each patch
    D2 = bsxfun(@plus, sum(patches.^2, 2), sum(W.^2, 2)') - 2 * (patches * W');
    [~, bmu] = min(D2, [], 2);

    % Count hits for each neuron
    hits = accumarray(bmu, 1, [K, 1]);
    [sorted_counts, sorted_idx] = sort(hits, 'descend');
    top_k = min(10, numel(sorted_idx));
    top_indices = sorted_idx(1:top_k);
    top_counts = sorted_counts(1:top_k);

    % Visualization
    figure('Color', 'w', 'Position', [100, 100, 700, 600]);
    set(gcf, 'ToolBar', 'none', 'MenuBar', 'none');
    hit_grid = reshape(hits, gs);
    vis_grid = log1p(hit_grid);  % Compress dynamic range for visibility
    imagesc(vis_grid);
    axis image off;
    colormap(hot);
    cb = colorbar;
    set(cb, 'Color', 'k');
    ticks = get(cb, 'Ticks');
    set(cb, 'TickLabels', arrayfun(@(t) sprintf('%.0f', exp(t) - 1), ticks, 'UniformOutput', false));
    ylabel(cb, 'Patch Count', 'Color', 'k');
    title('SOM Hit Map', 'FontSize', 12, 'FontWeight', 'bold');

    if nargin > 2 && ~isempty(outpath)
        [save_dir, save_name, ~] = fileparts(outpath);
        saveas(gcf, outpath);
        % Dump raw hit statistics for debugging/reporting
        stats_path = fullfile(save_dir, [save_name '_hits.csv']);
        neuron_coords = som.grid_coords;
        T = table((1:K)', neuron_coords(:,1), neuron_coords(:,2), hits, ...
            'VariableNames', {'neuron_idx','row','col','hit_count'});
        writetable(T, stats_path);
        close(gcf);
    end

    % Console summary for quick inspection
    fprintf('Top %d SOM neurons by hit count:\n', top_k);
    for i = 1:top_k
        fprintf('  #%d (idx=%d, coord=[%d,%d]) : %.0f hits\n', i, top_indices(i), ...
            som.grid_coords(top_indices(i),1), som.grid_coords(top_indices(i),2), top_counts(i));
    end

    % Optional output struct
    if nargout > 0
        hit_stats = struct('hits', hits, ...
                           'grid_hits', hit_grid, ...
                           'top_indices', top_indices, ...
                           'top_counts', top_counts);
    end
end
