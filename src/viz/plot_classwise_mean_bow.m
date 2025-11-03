function plot_classwise_mean_bow(features, labels, class_names, somK, outdir)
% PLOT_CLASSWISE_MEAN_BOW Plot mean BoW histogram for each class
%
% Inputs:
%   features    - [N x D] BoW feature matrix
%   labels      - [N x 1] class labels
%   class_names - Cell array of class name strings
%   somK        - Number of visual words (SOM neurons)
%   outdir      - Output directory for saving figures
%
% Theory: Class-average BoW shows typical visual word distribution.
%         Comparing confused classes reveals overlapping patterns.

    C = numel(class_names);

    for c = 1:C
        idx = labels == c;
        if ~any(idx)
            continue;
        end

        % Extract global histogram (first somK dimensions for spatial pyramid)
        if size(features, 2) > somK
            Hc = features(idx, 1:somK);
        else
            Hc = features(idx, :);
        end

        % Compute mean histogram across all samples
        m = mean(Hc, 1);

        % Visualization
        figure('Color', 'w', 'Position', [100, 100, 900, 450]);
        set(gcf, 'ToolBar', 'none', 'MenuBar', 'none');
        bar(1:length(m), m, 'FaceColor', [0.3, 0.5, 0.7], 'EdgeColor', 'none');
        set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
        xlim([0, length(m) + 1]);
        grid on;
        set(gca, 'GridColor', [0.8, 0.8, 0.8]);
        xlabel('Visual Word Index', 'FontSize', 10, 'Color', 'k');
        ylabel('Mean Feature Value', 'FontSize', 10, 'Color', 'k');
        title(sprintf('Mean BoW Histogram - Class %s', class_names{c}), ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', 'black');

        saveas(gcf, fullfile(outdir, sprintf('bow_mean_%s.png', class_names{c})));
        close(gcf);
    end
end
