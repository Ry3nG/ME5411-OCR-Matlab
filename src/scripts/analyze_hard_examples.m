% ANALYZE_HARD_EXAMPLES Visualize hard examples with smallest SVM margins
%
% This script loads Task 7.2 results and displays the hardest examples
% (those with smallest decision margins) to understand model weaknesses.
%
% ME5411 Robot Vision and AI - Hard Example Analysis

function analyze_hard_examples()
    % Load results from Task 7.2
    results_file = 'output/task7_2/results.mat';
    if ~exist(results_file, 'file')
        error('Results file not found: %s\nPlease run task7_2.m first', results_file);
    end

    load(results_file, 'results');
    load('data/test.mat', 'data_test', 'labels_test');

    % Class names
    class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

    % Output directory
    outdir = 'output/task7_2';

    fprintf('Analyzing hard examples...\n');

    %% 1. Find top-12 smallest margin examples
    margins = results.margins;
    predictions = results.predictions;
    true_labels = labels_test + 1;  % 1-based

    [sorted_margins, ord] = sort(margins, 'ascend');
    n_display = min(12, numel(margins));
    hard_idx = ord(1:n_display);

    fprintf('  Found %d hard examples (smallest margins)\n', n_display);

    %% 2. Visualize hard examples
    figure('Color', 'w', 'Position', [100, 100, 1200, 800]);
    for i = 1:n_display
        k = hard_idx(i);
        img = squeeze(data_test(:, :, 1, k));
        true_label = true_labels(k);
        pred_label = predictions(k);
        margin = margins(k);

        subplot(3, 4, i);
        imshow(img);
        hold on;

        % Color: red if wrong, green if correct
        if true_label ~= pred_label
            color = [1, 0, 0];  % Red
        else
            color = [0, 0.8, 0];  % Green
        end

        % Text annotation
        text(2, 10, sprintf('T:%s P:%s\nm=%.3f', ...
                            class_names{true_label}, ...
                            class_names{pred_label}, ...
                            margin), ...
             'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', [0, 0, 0, 0.6], 'EdgeColor', color, 'LineWidth', 2);
        hold off;
    end

    sgtitle('Hard Examples (Smallest SVM Margins)', ...
            'FontSize', 14, 'FontWeight', 'bold');

    saveas(gcf, fullfile(outdir, 'hard_examples_margins.png'));
    close(gcf);

    fprintf('  Saved: hard_examples_margins.png\n');

    %% 3. Margin distribution by class
    fprintf('  Analyzing margin distribution by class...\n');
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);

    for c = 1:7
        idx = true_labels == c;
        class_margins = margins(idx);

        subplot(2, 4, c);
        histogram(class_margins, 20, 'FaceColor', [0.3, 0.5, 0.7], 'EdgeColor', 'k');
        hold on;
        xline(0, 'r--', 'LineWidth', 2);  % Decision boundary
        hold off;
        grid on;
        xlabel('Margin', 'FontSize', 10);
        ylabel('Count', 'FontSize', 10);
        title(sprintf('Class: %s', class_names{c}), 'FontSize', 11, 'FontWeight', 'bold');
        xlim([-3, 5]);
    end

    sgtitle('SVM Decision Margins by Class', 'FontSize', 14, 'FontWeight', 'bold');

    saveas(gcf, fullfile(outdir, 'margins_by_class.png'));
    close(gcf);

    fprintf('  Saved: margins_by_class.png\n');

    %% 4. Summary statistics
    fprintf('\n  === Hard Example Summary ===\n');
    fprintf('  Smallest margin: %.4f (idx=%d, T=%s, P=%s)\n', ...
            sorted_margins(1), hard_idx(1), ...
            class_names{true_labels(hard_idx(1))}, ...
            class_names{predictions(hard_idx(1))});

    % Count misclassifications among hard examples
    n_wrong = sum(true_labels(hard_idx) ~= predictions(hard_idx));
    fprintf('  Misclassifications in top-%d hard: %d (%.1f%%)\n', ...
            n_display, n_wrong, n_wrong/n_display*100);

    % Most confused pairs
    fprintf('\n  Most confused class pairs (among hard examples):\n');
    conf_pairs = [];
    for i = 1:n_display
        k = hard_idx(i);
        if true_labels(k) ~= predictions(k)
            conf_pairs = [conf_pairs; true_labels(k), predictions(k)]; %#ok<AGROW>
        end
    end

    if ~isempty(conf_pairs)
        [unique_pairs, ~, ic] = unique(conf_pairs, 'rows');
        counts = accumarray(ic, 1);
        [~, ord] = sort(counts, 'descend');

        for i = 1:min(3, size(unique_pairs, 1))
            pair = unique_pairs(ord(i), :);
            fprintf('    %s -> %s: %d times\n', ...
                    class_names{pair(1)}, class_names{pair(2)}, counts(ord(i)));
        end
    end

    fprintf('\nAnalysis complete!\n');
end
