%% Compare Original vs Improved Task 7.2 Implementation
% Analyzes and visualizes differences between two implementations

clear; clc; close all;

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
cd(project_root);

output_dir = 'output/comparison';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load Results
fprintf('Loading results...\n');

% Original version
if exist('output/task7_2/results.mat', 'file')
    load('output/task7_2/results.mat', 'results');
    res_orig = results;
    has_orig = true;
    fprintf('  Original: Accuracy = %.2f%%\n', res_orig.accuracy * 100);
else
    has_orig = false;
    fprintf('  Original results not found\n');
end

% Improved version
if exist('output/task7_2_improved/results.mat', 'file')
    load('output/task7_2_improved/results.mat', 'results');
    res_improved = results;
    has_improved = true;
    fprintf('  Improved: Accuracy = %.2f%%\n', res_improved.accuracy * 100);
else
    has_improved = false;
    fprintf('  Improved results not found\n');
end

if ~has_orig && ~has_improved
    error('No results found to compare');
end

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = 7;

%% Compare Accuracy
if has_orig && has_improved
    fprintf('\n=== Accuracy Comparison ===\n');
    fprintf('Original:  %.2f%%\n', res_orig.accuracy * 100);
    fprintf('Improved:  %.2f%%\n', res_improved.accuracy * 100);
    fprintf('Difference: %+.2f%%\n', ...
        (res_improved.accuracy - res_orig.accuracy) * 100);

    % Per-class comparison
    fprintf('\n=== Per-Class Accuracy ===\n');
    fprintf('Class\tOriginal\tImproved\tDiff\n');
    for c = 1:num_classes
        fprintf('%s\t%.2f%%\t\t%.2f%%\t\t%+.2f%%\n', ...
            class_names{c}, ...
            res_orig.class_accuracy(c) * 100, ...
            res_improved.class_accuracy(c) * 100, ...
            (res_improved.class_accuracy(c) - res_orig.class_accuracy(c)) * 100);
    end

    % Time comparison
    fprintf('\n=== Time Comparison ===\n');
    fprintf('Training:\n');
    fprintf('  Original: %.2f min\n', res_orig.training_time / 60);
    fprintf('  Improved: %.2f min\n', res_improved.training_time / 60);
    fprintf('  Speedup: %.2fx\n', res_orig.training_time / res_improved.training_time);

    fprintf('Testing:\n');
    fprintf('  Original: %.2f s\n', res_orig.test_time);
    fprintf('  Improved: %.2f s\n', res_improved.test_time);
    fprintf('  Speedup: %.2fx\n', res_orig.test_time / res_improved.test_time);

    %% Visualize Per-Class Comparison
    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 500]);

    subplot(1, 2, 1);
    x = 1:num_classes;
    bar(x - 0.2, res_orig.class_accuracy * 100, 0.4, 'FaceColor', [0.8, 0.4, 0.4]);
    hold on;
    bar(x + 0.2, res_improved.class_accuracy * 100, 0.4, 'FaceColor', [0.4, 0.6, 0.8]);
    hold off;
    xlabel('Class', 'FontWeight', 'bold');
    ylabel('Accuracy (%)', 'FontWeight', 'bold');
    title('Per-Class Accuracy Comparison', 'FontWeight', 'bold');
    xticks(1:num_classes);
    xticklabels(class_names);
    legend({'Original', 'Improved'}, 'Location', 'best');
    grid on;
    ylim([0, 100]);

    subplot(1, 2, 2);
    metrics = {'Overall\nAccuracy', 'Training\nTime', 'Test\nTime'};
    orig_vals = [res_orig.accuracy * 100, res_orig.training_time / 60, res_orig.test_time];
    impr_vals = [res_improved.accuracy * 100, res_improved.training_time / 60, res_improved.test_time];
    % Normalize for visualization
    orig_vals_norm = [orig_vals(1), orig_vals(2) / max(orig_vals(2:3)) * 100, ...
                      orig_vals(3) / max(orig_vals(2:3)) * 100];
    impr_vals_norm = [impr_vals(1), impr_vals(2) / max(orig_vals(2:3)) * 100, ...
                      impr_vals(3) / max(orig_vals(2:3)) * 100];

    x = 1:3;
    bar(x - 0.2, orig_vals_norm, 0.4, 'FaceColor', [0.8, 0.4, 0.4]);
    hold on;
    bar(x + 0.2, impr_vals_norm, 0.4, 'FaceColor', [0.4, 0.6, 0.8]);
    hold off;
    xlabel('Metric', 'FontWeight', 'bold');
    ylabel('Normalized Value', 'FontWeight', 'bold');
    title('Overall Metrics Comparison', 'FontWeight', 'bold');
    xticks(1:3);
    xticklabels(metrics);
    legend({'Original', 'Improved'}, 'Location', 'best');
    grid on;

    saveas(fig, fullfile(output_dir, 'comparison.png'));
    close(fig);

    fprintf('\nComparison plot saved to %s\n', output_dir);
end

%% Analyze Confusion Patterns
if has_improved
    fprintf('\n=== Confusion Analysis (Improved) ===\n');
    CM = res_improved.confusion_matrix;
    CM_norm = CM ./ sum(CM, 2);

    % Find top confusions (excluding diagonal)
    CM_off = CM_norm;
    for i = 1:num_classes
        CM_off(i, i) = 0;
    end

    [sorted_conf, idx] = sort(CM_off(:), 'descend');
    fprintf('Top 5 confusion pairs:\n');
    for k = 1:min(5, length(sorted_conf))
        if sorted_conf(k) > 0.01
            [i, j] = ind2sub(size(CM_off), idx(k));
            fprintf('  %s -> %s: %.1f%% (%d samples)\n', ...
                class_names{i}, class_names{j}, ...
                sorted_conf(k) * 100, CM(i, j));
        end
    end
end

fprintf('\nDone.\n');
