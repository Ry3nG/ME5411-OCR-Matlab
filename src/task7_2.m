%% Task 7.2 Final: SOM (Batch) + BoW + SVM with Enhanced Visualizations
% ME5411 OCR Project
%
% Improvements:
% 1. Mini-batch SOM training (faster, maintains accuracy)
% 2. Enhanced visualizations (U-Matrix, Hit Map, Margins, etc.)
% 3. Code organization and documentation
%
% Theory: All methods from Part-II lectures (non-CNN)

clear; clc; close all;
rng(0, 'twister');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/viz'));
output_dir = 'output/task7_2';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('=== Task 7.2: SOM + BoW + SVM Classification ===\n\n');

%% Configuration
cfg = struct();
cfg.patch_size = 8;
cfg.num_patch_samples = 100000;
cfg.som_grid = [10, 10];
cfg.som_iterations = 50000;
cfg.som_lr_init = 0.5;
cfg.som_lr_final = 0.01;
cfg.som_sigma_init = max(cfg.som_grid) / 2;
cfg.som_sigma_final = 0.5;
cfg.som_batch = 32;
cfg.stride = 4;
cfg.soft_voting = true;
cfg.sigma_bow = 0.75;
cfg.spatial_pyramid = true;
cfg.C = 1.0;
cfg.svm_epochs = 200;
cfg.svm_lr = 0.01;
cfg.use_pca = true;
cfg.pca_var = 0.95;
cfg.plot_per_class_accuracy = false;
cfg.min_patch_std = 0.001;

fprintf('Configuration:\n');
disp(cfg);
fprintf('\n');

%% Load Data
fprintf('Loading data...\n');
load('data/train.mat', 'data_train', 'labels_train');
load('data/test.mat', 'data_test', 'labels_test');

Xtr = data_train;
Ytr = labels_train + 1;
Xte = data_test;
Yte = labels_test + 1;

[H, W, ~, N_train] = size(Xtr);
N_test = size(Xte, 4);
num_classes = 7;
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

fprintf('Training: %d samples, Test: %d samples\n', N_train, N_test);
fprintf('Image size: %dx%d, Classes: %d\n\n', H, W, num_classes);

%% Stage 1: Extract Patches
fprintf('=== Stage 1: Patch Extraction ===\n');
tic;
patches = extract_patches(Xtr, Ytr, cfg.patch_size, cfg.num_patch_samples, ...
    'normalize', true, 'verbose', true, 'min_patch_std', cfg.min_patch_std);
t_patches = toc;
fprintf('Time: %.2f seconds\n\n', t_patches);

%% Stage 2: Train SOM (Mini-batch)
fprintf('=== Stage 2: SOM Training (Mini-batch) ===\n');
tic;
som = train_som_batch(patches, cfg.som_grid, cfg.som_iterations, ...
    'lr_init', cfg.som_lr_init, 'lr_final', cfg.som_lr_final, ...
    'sigma_init', cfg.som_sigma_init, 'sigma_final', cfg.som_sigma_final, ...
    'batch', cfg.som_batch, 'verbose', true);
t_som = toc;
fprintf('Time: %.2f seconds\n\n', t_som);

save(fullfile(output_dir, 'som_model.mat'), 'som');

%% Stage 3: Extract BoW Features
fprintf('=== Stage 3: BoW Feature Extraction ===\n');
fprintf('Extracting training features...\n');
tic;
Ftr = extract_bow_features(Xtr, som, cfg.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
    'spatial_pyramid', cfg.spatial_pyramid, 'verbose', true, ...
    'min_patch_std', cfg.min_patch_std);
t_train_feat = toc;

fprintf('Extracting test features...\n');
tic;
Fte = extract_bow_features(Xte, som, cfg.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
    'spatial_pyramid', cfg.spatial_pyramid, 'verbose', true, ...
    'min_patch_std', cfg.min_patch_std);
t_test_feat = toc;

fprintf('Feature dimension (pre-PCA): %d\n', size(Ftr, 2));
fprintf('Training features: %.2f seconds\n', t_train_feat);
fprintf('Test features: %.2f seconds\n\n', t_test_feat);

if cfg.use_pca
    fprintf('Applying PCA (retain %.0f%% variance)...\n', cfg.pca_var * 100);
    [Ftr, Fte, pca_model] = apply_pca(Ftr, Fte, cfg.pca_var);
    fprintf('Feature dimension after PCA: %d\n\n', size(Ftr, 2));
    save(fullfile(output_dir, 'pca_model.mat'), 'pca_model');
else
    fprintf('PCA disabled; using full feature dimension.\n\n');
end

%% Stage 4: Train SVM
fprintf('=== Stage 4: SVM Training ===\n');
tic;
models = trainMulticlassSVM(Ftr, Ytr, num_classes, cfg.C, ...
    'max_epochs', cfg.svm_epochs, 'lr', cfg.svm_lr, 'verbose', false);
t_svm = toc;
fprintf('Time: %.2f seconds\n\n', t_svm);

save(fullfile(output_dir, 'svm_models.mat'), 'models');

t_total_train = t_patches + t_som + t_train_feat + t_svm;
fprintf('=== Training Summary ===\n');
fprintf('Total training time: %.2f seconds (%.2f minutes)\n\n', ...
    t_total_train, t_total_train / 60);

%% Stage 5: Evaluation
fprintf('=== Stage 5: Evaluation ===\n');
tic;
predictions = predictMulticlassSVM(models, Fte);
t_predict = toc;

accuracy = mean(predictions == Yte);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Prediction time: %.2f seconds\n', t_predict);
fprintf('Per-sample time: %.4f seconds\n\n', t_predict / N_test);

%% Compute Metrics
conf_matrix = zeros(num_classes, num_classes);
for i = 1:N_test
    conf_matrix(Yte(i), predictions(i)) = conf_matrix(Yte(i), predictions(i)) + 1;
end

class_accuracy = diag(conf_matrix) ./ sum(conf_matrix, 2);
fprintf('=== Per-Class Accuracy ===\n');
for c = 1:num_classes
    fprintf('Class %d (%s): %.2f%%\n', c, class_names{c}, class_accuracy(c) * 100);
end
fprintf('\n');

%% Stage 6: Visualizations
fprintf('=== Stage 6: Generating Visualizations ===\n');

% 1. SOM Codebook
fprintf('1. SOM codebook...\n');
fig = figure('Position', [100, 100, 1000, 1000], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
for i = 1:som.num_neurons
    subplot(cfg.som_grid(1), cfg.som_grid(2), i);
    patch_img = reshape(som.weights(i, :), [cfg.patch_size, cfg.patch_size]);
    patch_img = (patch_img - min(patch_img(:))) / (max(patch_img(:)) - min(patch_img(:)) + 1e-10);
    imshow(patch_img, []);
    axis off;
end
saveas(fig, fullfile(output_dir, 'som_codebook.png'));
close(fig);

% 2. U-Matrix
fprintf('2. U-Matrix...\n');
plot_som_umatrix(som, fullfile(output_dir, 'som_umatrix.png'));

% 3. Hit Map
fprintf('3. Hit Map...\n');
plot_som_hits(patches, som, fullfile(output_dir, 'som_hitmap.png'));

% 4. Confusion Matrix
fprintf('4. Confusion matrix...\n');
conf_matrix_norm = conf_matrix ./ sum(conf_matrix, 2);
fig = figure('Position', [100, 100, 900, 800], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(conf_matrix_norm);
colormap(flipud(gray));  % High values = dark, low values = light
cb = colorbar;
caxis([0, 1]);
set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
for i = 1:num_classes
    for j = 1:num_classes
        count = conf_matrix(i, j);
        percentage = conf_matrix_norm(i, j) * 100;
        % Use white text for dark cells (high values), black for light cells
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

% 5. Per-Class Accuracy (optional figure)
if isfield(cfg, 'plot_per_class_accuracy') && cfg.plot_per_class_accuracy
    fprintf('5. Per-class accuracy...\n');
    fig = figure('Position', [100, 100, 800, 500], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    bar(1:num_classes, class_accuracy * 100, 'FaceColor', [0.2, 0.4, 0.6]);
    set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    xticks(1:num_classes);
    xticklabels(class_names);
    ylim([0, 100]);
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);
    for c = 1:num_classes
        text(c, class_accuracy(c) * 100 + 2, sprintf('%.1f%%', class_accuracy(c) * 100), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'black');
    end
    saveas(fig, fullfile(output_dir, 'per_class_accuracy.png'));
    close(fig);
else
    fprintf('5. Per-class accuracy plot skipped (see confusion matrix / table).\n');
end

% 6. Class-wise Mean BoW
fprintf('6. Class-wise mean BoW histograms...\n');
plot_classwise_mean_bow(Fte, Yte, class_names, som.num_neurons, output_dir);

% 7. SVM Margins
fprintf('7. SVM margin distribution...\n');
[margins, ~] = multiclass_margins(models, Fte, Yte);
plot_svm_margins(margins, fullfile(output_dir, 'svm_margins.png'));

% 8. Misclassifications
fprintf('8. Misclassification examples (one per class)...\n');
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
            'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    end
    saveas(fig, fullfile(output_dir, 'misclassifications.png'));
    close(fig);
else
    fprintf('  All classes predicted correctly; no misclassifications to show.\n');
end

fprintf('All visualizations saved.\n\n');

%% Save Results
fprintf('=== Saving Results ===\n');
results = struct();
results.accuracy = accuracy;
results.confusion_matrix = conf_matrix;
results.class_accuracy = class_accuracy;
results.predictions = predictions;
results.margins = margins;
results.training_time = t_total_train;
results.test_time = t_test_feat + t_predict;
results.config = cfg;

save(fullfile(output_dir, 'results.mat'), 'results');
fprintf('Results saved.\n\n');

%% Final Summary
fprintf('=== Task 7.2 Complete ===\n');
fprintf('Method: SOM (Mini-batch) + Bag-of-Visual-Words + Linear SVM\n');
fprintf('Visual Codebook Size: %d\n', som.num_neurons);
fprintf('Feature Dimension: %d\n', size(Ftr, 2));
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Training Time: %.2f minutes\n', t_total_train / 60);
fprintf('Test Time: %.2f seconds\n', t_test_feat + t_predict);
fprintf('\nAll outputs saved to: %s\n', output_dir);
