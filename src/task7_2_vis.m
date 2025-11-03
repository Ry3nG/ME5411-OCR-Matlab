%% Task 7.2 Visualization Only
% Re-generate visualizations without re-training
% Load saved models and data, regenerate all figures

clear; clc; close all;
rng(0, 'twister');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/viz'));
output_dir = 'output/task7_2';

fprintf('=== Task 7.2: Regenerating Visualizations ===\n\n');

%% Load saved data
fprintf('Loading saved models and data...\n');
load('data/test.mat', 'data_test', 'labels_test');
load(fullfile(output_dir, 'som_model.mat'), 'som');
load(fullfile(output_dir, 'svm_models.mat'), 'models');
load(fullfile(output_dir, 'pca_model.mat'), 'pca_model');
load(fullfile(output_dir, 'results.mat'), 'results');

Xte = data_test;
Yte = labels_test + 1;
[H, W, ~, N_test] = size(Xte);
num_classes = 7;
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
cfg = results.config;

% Re-extract test features
fprintf('Re-extracting test features...\n');
Fte = extract_bow_features(Xte, som, cfg.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
    'spatial_pyramid', cfg.spatial_pyramid, 'verbose', false, ...
    'min_patch_std', cfg.min_patch_std);

% Apply PCA
Fte = (Fte - pca_model.mu) * pca_model.W;

% Predict
predictions = predictMulticlassSVM(models, Fte);
accuracy = mean(predictions == Yte);
fprintf('Test Accuracy: %.2f%%\n\n', accuracy * 100);

%% Confusion Matrix
fprintf('Generating confusion matrix...\n');
conf_matrix = zeros(num_classes, num_classes);
for i = 1:N_test
    conf_matrix(Yte(i), predictions(i)) = conf_matrix(Yte(i), predictions(i)) + 1;
end

conf_matrix_norm = conf_matrix ./ sum(conf_matrix, 2);
fig = figure('Position', [100, 100, 900, 800], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(conf_matrix_norm);
colormap(flipud(gray));
cb = colorbar;
caxis([0, 1]);
set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
set(cb, 'Color', 'black');
for i = 1:num_classes
    for j = 1:num_classes
        count = conf_matrix(i, j);
        percentage = conf_matrix_norm(i, j) * 100;
        if conf_matrix_norm(i, j) > 0.5
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', text_color, 'FontSize', 11);
    end
end
xlabel('Predicted Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('True Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
xticks(1:num_classes);
xticklabels(class_names);
yticks(1:num_classes);
yticklabels(class_names);
axis square;
grid off;
saveas(fig, fullfile(output_dir, 'confusion_matrix.png'));
close(fig);
fprintf('Saved: confusion_matrix.png\n');

%% SVM Margins (IMPROVED)
fprintf('Generating SVM margin distribution...\n');
[margins, ~] = multiclass_margins(models, Fte, Yte);

fig = figure('Color', 'w', 'Position', [100, 100, 900, 600]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

histogram(margins, 50, 'FaceColor', [0.2, 0.4, 0.6], 'EdgeColor', 'none');
grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8, 0.8, 0.8]);
xlabel('Margin (True Class Score - Second Best Score)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('Number of Samples', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
title('SVM Margin Distribution', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'black');

% Add vertical line at margin=0
hold on;
yl = ylim;
plot([0, 0], yl, 'r-', 'LineWidth', 2.5);

% Add shaded regions and annotations
x_neg = xlim;
% Misclassified region (margin < 0)
patch([x_neg(1), 0, 0, x_neg(1)], [0, 0, yl(2), yl(2)], ...
    [1, 0.9, 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% Correctly classified region (margin > 0)
patch([0, x_neg(2), x_neg(2), 0], [0, 0, yl(2), yl(2)], ...
    [0.9, 1, 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Annotations
text(-0.5, yl(2) * 0.92, 'Misclassified', ...
    'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', [0.6, 0, 0], 'FontWeight', 'bold');
text(0.5, yl(2) * 0.92, 'Correctly Classified', ...
    'HorizontalAlignment', 'left', 'FontSize', 10, 'Color', [0, 0.5, 0], 'FontWeight', 'bold');

% Statistics annotation
num_correct = sum(margins > 0);
num_incorrect = sum(margins <= 0);
mean_margin = mean(margins);
text(0.98, 0.95, sprintf('Correct: %d (%.1f%%)\nIncorrect: %d (%.1f%%)\nMean margin: %.2f', ...
    num_correct, num_correct/length(margins)*100, ...
    num_incorrect, num_incorrect/length(margins)*100, ...
    mean_margin), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black', 'Margin', 5);

hold off;

saveas(fig, fullfile(output_dir, 'svm_margins.png'));
close(fig);
fprintf('Saved: svm_margins.png\n');

%% Misclassification Examples
fprintf('Generating misclassification examples...\n');
example_idx = [];
for c = 1:num_classes
    idx = find((Yte == c) & (predictions ~= c), 1, 'first');
    if ~isempty(idx)
        example_idx(end+1) = idx; %#ok<AGROW>
    end
end

if ~isempty(example_idx)
    fig = figure('Position', [100, 100, 260 * numel(example_idx), 280], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    tiledlayout(1, numel(example_idx), 'Padding', 'compact', 'TileSpacing', 'compact');
    for i = 1:numel(example_idx)
        idx = example_idx(i);
        nexttile;
        imshow(squeeze(Xte(:, :, 1, idx)));
        title(sprintf('T:%s  P:%s', class_names{Yte(idx)}, class_names{predictions(idx)}), ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', 'black');
    end
    saveas(fig, fullfile(output_dir, 'misclassifications.png'));
    close(fig);
    fprintf('Saved: misclassifications.png\n');
end

%% Class-wise Mean BoW Histograms
fprintf('Generating class-wise mean BoW histograms...\n');
plot_classwise_mean_bow(Fte, Yte, class_names, som.num_neurons, output_dir);
fprintf('Saved: bow_mean_*.png (7 files)\n');

%% Complete
fprintf('\n=== Visualization Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
