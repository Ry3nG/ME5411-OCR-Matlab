%% Task 7.1 Visualization Only
% Re-generate all visualizations without re-training
% Load saved model and data, regenerate all figures for the report

clear; clc; close all;
rng(0, 'twister');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Locate most recent Task 7.1 run directory
output_base = fullfile('output', 'task7_1');
if ~exist(output_base, 'dir')
    error('Task 7.1 output directory not found: %s', output_base);
end

subdirs = dir(fullfile(output_base, '*-*_*-*-*'));
subdirs = subdirs([subdirs.isdir]);
if isempty(subdirs)
    error('No timestamped runs found in %s', output_base);
end
[~, sort_idx] = sort([subdirs.datenum], 'descend');
model_dir = fullfile(output_base, subdirs(sort_idx(1)).name);

output_dir = fullfile(model_dir, 'figures_regenerated');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('=== Task 7.1: Regenerating All Visualizations ===\n\n');

%% Load saved data
fprintf('Loading saved model and data...\n');

cnn_path_candidates = {
    fullfile(model_dir, 'cnn_final.mat'), ...
    fullfile(model_dir, 'cnn_best_acc.mat'), ...
    fullfile(model_dir, 'cnn.mat')
};

cnn = [];
for i = 1:numel(cnn_path_candidates)
    if exist(cnn_path_candidates{i}, 'file')
        tmp = load(cnn_path_candidates{i}, 'cnn');
        cnn = tmp.cnn;
        break;
    end
end

if isempty(cnn)
    error('No CNN model file found in %s', model_dir);
end

preds_file = fullfile(model_dir, 'predictions.mat');
if exist(preds_file, 'file')
    preds_data = load(preds_file);
    if isfield(preds_data, 'preds_test')
        preds = preds_data.preds_test;
        labels_test_eval = preds_data.labels_test_eval;
    elseif isfield(preds_data, 'preds')
        preds = preds_data.preds;
        labels_test_eval = preds_data.labels_test_eval;
    else
        error('Unexpected contents in %s', preds_file);
    end
else
    error('predictions.mat not found in %s', model_dir);
end

acc_train = []; acc_test = []; loss_ar = []; lr_ar = [];
acc_train_path = fullfile(model_dir, 'acc_train.mat');
acc_test_path = fullfile(model_dir, 'acc_test.mat');
loss_path = fullfile(model_dir, 'loss_ar.mat');
lr_path = fullfile(model_dir, 'lr_ar.mat');

if exist(acc_train_path, 'file')
    acc_train = load(acc_train_path, 'acc_train');
    acc_train = acc_train.acc_train;
end
if exist(acc_test_path, 'file')
    acc_test = load(acc_test_path, 'acc_test');
    acc_test = acc_test.acc_test;
end
if exist(loss_path, 'file')
    loss_ar = load(loss_path, 'loss_ar');
    loss_ar = loss_ar.loss_ar;
end
if exist(lr_path, 'file')
    lr_ar = load(lr_path, 'lr_ar');
    lr_ar = lr_ar.lr_ar;
end

load('data/test.mat', 'data_test', 'labels_test');
load('data/train.mat', 'data_train', 'labels_train');

% Convert labels (dataset is 0-indexed, MATLAB uses 1-indexed)
labels_test = labels_test + 1;
labels_train = labels_train + 1;

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

fprintf('Model loaded successfully.\n');
fprintf('Test samples: %d\n', size(data_test, 4));
fprintf('Train samples: %d\n\n', size(data_train, 4));

%% 1. Dataset Samples Visualization
fprintf('Generating class samples visualization...\n');

% Visualization config
invert_input_polarity = false; % show as black background, white glyph (default dataset polarity)

% Select fewer samples from each class for brevity
num_samples_per_class = 7;
fig = figure('Position', [100, 100, 1400, 900], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

% Build a tight grid (minimal gaps) using manual axes positions
gap = 0.0;                  % gap between tiles (normalized)
margin_h = 0.005;           % top/bottom margin
margin_w = 0.005;           % left margin
margin_w_right = 0.005;     % right margin
tile_w = (1 - margin_w - margin_w_right - (num_samples_per_class-1)*gap) / num_samples_per_class;
num_displayed_classes = 4;
tile_h = (1 - 2*margin_h - (num_displayed_classes-1)*gap) / num_displayed_classes;

for c = 0:min(3, num_classes-1)
    idx = find(labels_train == c+1);
    % Select 10 evenly spaced samples
    selected_idx = idx(round(linspace(1, length(idx), num_samples_per_class)));

    for s = 1:num_samples_per_class
        left = margin_w + (s-1) * (tile_w + gap);
        bottom = 1 - margin_h - (c+1) * tile_h - c * gap;

        img = data_train(:,:,1,selected_idx(s));
        if invert_input_polarity
            img = 1 - img;
        end

        ax = axes('Parent', fig, 'Position', [left, bottom, tile_w, tile_h], ...
            'Units', 'normalized');
        imshow(img, []);
        axis(ax, 'off');
        set(ax, 'LooseInset', [0, 0, 0, 0]);
        set(ax, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none');
    end
end

saveas(fig, fullfile(output_dir, 'class_samples.png'));
close(fig);
fprintf('Saved: class_samples.png\n');

%% 2. Training Curves
if ~isempty(acc_train) && ~isempty(acc_test) && ~isempty(loss_ar) && ~isempty(lr_ar)
    fprintf('Generating training curves...\n');

    fig = figure('Position', [100, 100, 1200, 400], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

    % Accuracy curve
    subplot(1, 3, 1);
    epochs = 1:length(acc_train);
    plot(epochs, acc_train * 100, 'b-', 'LineWidth', 2); hold on;
    plot(epochs, acc_test * 100, 'r-', 'LineWidth', 2);
    set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    leg = legend({'Train', 'Test'}, 'Location', 'southeast', 'TextColor', 'black', 'EdgeColor', 'black');
    set(leg, 'Color', 'white');
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);

    % Loss curve
    subplot(1, 3, 2);
    plot(1:length(loss_ar), loss_ar, 'k-', 'LineWidth', 1.5);
    set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);

    % Learning rate schedule
    subplot(1, 3, 3);
    plot(1:length(lr_ar), lr_ar, 'g-', 'LineWidth', 2);
    set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Learning Rate', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);

    saveas(fig, fullfile(output_dir, 'training_curves.png'));
    close(fig);
    fprintf('Saved: training_curves.png\n');
else
    fprintf('Training curves unavailable (no acc/loss logs found). Skipping.\n');
end

%% 3. Confusion Matrix
fprintf('Generating confusion matrix...\n');

% Re-predict to ensure consistency
[preds_new, ~] = predict(cnn, data_test);
preds_new = preds_new - 1;
labels_test_eval = labels_test - 1;

confMat = zeros(num_classes);
for i = 1:length(preds_new)
    true_label = labels_test_eval(i) + 1;
    pred_label = preds_new(i) + 1;
    confMat(true_label, pred_label) = confMat(true_label, pred_label) + 1;
end

% Normalized confusion matrix
confMat_norm = confMat ./ sum(confMat, 2);

fig = figure('Position', [100, 100, 900, 800], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(confMat_norm);
colormap(flipud(gray));
cb = colorbar;
caxis([0, 1]);
set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
set(cb, 'Color', 'black');
xlabel('Predicted Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('True Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
set(gca, 'XTick', 1:num_classes, 'XTickLabel', class_names);
set(gca, 'YTick', 1:num_classes, 'YTickLabel', class_names);
axis square;
grid off;

% Add text annotations
for i = 1:num_classes
    for j = 1:num_classes
        count = confMat(i, j);
        percentage = confMat_norm(i, j) * 100;

        if confMat_norm(i, j) > 0.5
            textColor = 'white';
        else
            textColor = 'black';
        end

        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', textColor, 'FontSize', 11);
    end
end

saveas(fig, fullfile(output_dir, 'confusion_matrix.png'));
close(fig);
fprintf('Saved: confusion_matrix.png\n');

%% 4. Correctly Classified Samples
fprintf('Generating correctly classified samples...\n');

fig = figure('Position', [100, 100, 1400, 200], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

for c = 0:num_classes-1
    idx = find((labels_test_eval == c) & (preds_new == c));
    if length(idx) >= 1
        % Select sample with median confidence (interesting but correct)
        class_samples = data_test(:,:,1,idx);
        confidences = zeros(length(idx), 1);
        for i = 1:length(idx)
            predict(cnn, data_test(:,:,1,idx(i)));
            confidences(i) = cnn.layers{end}.activations(c+1);
        end

        [~, sorted_idx] = sort(confidences);
        select_position = max(1, round(length(sorted_idx) * 0.35));
        selected_idx = idx(sorted_idx(select_position));

        subplot(1, 7, c+1);
        img = data_test(:,:,1,selected_idx);
        if invert_input_polarity
            img = 1 - img;
        end
        imshow(img, []);
        title(sprintf('Class %s', class_names{c+1}), 'FontSize', 10, 'Color', 'black');
        axis off;
    end
end

saveas(fig, fullfile(output_dir, 'correct_predictions.png'));
close(fig);
fprintf('Saved: correct_predictions.png\n');

%% 5. Misclassification Examples
fprintf('Generating misclassification examples...\n');

misclass_idx = find(preds_new ~= labels_test_eval);
if length(misclass_idx) > 0
    fig = figure('Position', [100, 100, 1400, 200], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    plot_idx = 1;

    for c = 0:num_classes-1
        idx = find((labels_test_eval == c) & (preds_new ~= c));
        if length(idx) >= 1
            subplot(1, 7, plot_idx);
            sample_idx = idx(1);
            img = data_test(:,:,1,sample_idx);
            if invert_input_polarity
                img = 1 - img;
            end
            imshow(img, []);
            true_class = class_names{labels_test_eval(sample_idx) + 1};
            pred_class = class_names{preds_new(sample_idx) + 1};
            title(sprintf('%s→%s', true_class, pred_class), 'FontSize', 10, 'Color', 'red');
            axis off;
            plot_idx = plot_idx + 1;
        end
    end

    saveas(fig, fullfile(output_dir, 'misclassifications.png'));
    close(fig);
    fprintf('Saved: misclassifications.png\n');
end

%% 6. Feature Evolution Visualization
fprintf('Generating feature evolution visualization...\n');

% Select one correctly classified digit '8'
correct_idx = find((labels_test_eval == 3) & (preds_new == 3), 1);
if ~isempty(correct_idx)
    sample_img = data_test(:,:,1,correct_idx);

    % Forward pass to extract activations
    cnn_temp = cnn;
    local_option.train_mode = false;
    cnn_temp = forward(cnn_temp, sample_img, local_option);

    conv1_activations = cnn_temp.layers{2}.activations;
    conv2_activations = cnn_temp.layers{3}.activations;
    conv3_activations = cnn_temp.layers{4}.activations;

    % Save input image
    img = sample_img;
    if invert_input_polarity
        img = 1 - img;
    end
    % Convert to uint8 and save at original 64x64 size
    img_uint8 = uint8(255 * img);
    imwrite(img_uint8, fullfile(output_dir, 'feature_input.png'));

    % Conv1 feature maps (dynamic grid)
    [~, ~, conv1_channels] = size(conv1_activations);
    conv1_cols = ceil(sqrt(conv1_channels));
    conv1_rows = ceil(conv1_channels / conv1_cols);
    fig = figure('Position', [100, 100, 400, 400], 'Color', 'white');
    for i = 1:conv1_channels
        subplot(conv1_rows, conv1_cols, i);
        imshow(conv1_activations(:,:,i), []);
        axis off;
    end
    set(gcf, 'Color', 'white');
    saveas(fig, fullfile(output_dir, 'feature_conv1.png'));
    close(fig);

    % Conv2 feature maps (dynamic grid)
    [~, ~, conv2_channels] = size(conv2_activations);
    conv2_cols = ceil(sqrt(conv2_channels));
    conv2_rows = ceil(conv2_channels / conv2_cols);
    fig = figure('Position', [100, 100, 600, 600], 'Color', 'white');
    for i = 1:conv2_channels
        subplot(conv2_rows, conv2_cols, i);
        imshow(conv2_activations(:,:,i), []);
        axis off;
    end
    set(gcf, 'Color', 'white');
    saveas(fig, fullfile(output_dir, 'feature_conv2.png'));
    close(fig);

    % Conv3 feature maps (dynamic grid)
    [~, ~, conv3_channels] = size(conv3_activations);
    conv3_cols = ceil(sqrt(conv3_channels));
    conv3_rows = ceil(conv3_channels / conv3_cols);
    fig = figure('Position', [100, 100, 800, 800], 'Color', 'white');
    for i = 1:conv3_channels
        subplot(conv3_rows, conv3_cols, i);
        imshow(conv3_activations(:,:,i), []);
        axis off;
    end
    set(gcf, 'Color', 'white');
    saveas(fig, fullfile(output_dir, 'feature_conv3.png'));
    close(fig);

    fprintf('Saved: feature_input.png, feature_conv1.png, feature_conv2.png, feature_conv3.png\n');
end

%% 7. Misclassification Feature Map Comparison
fprintf('Generating misclassification comparison...\n');

% Find D->4 confusion (most common)
true_class = 5;   % D (0-indexed)
pred_class = 1;   % 4 (0-indexed)

misclass_idx = find((labels_test_eval == true_class) & (preds_new == pred_class), 1);
if ~isempty(misclass_idx)
    % Find correctly classified sample from predicted class
    correct_idx = find((labels_test_eval == pred_class) & (preds_new == pred_class), 1);

    % Extract feature maps for both samples
    local_option.train_mode = false;

    % Misclassified sample
    sample_mis = data_test(:,:,1,misclass_idx);
    cnn_mis = cnn;
    cnn_mis = forward(cnn_mis, sample_mis, local_option);
    conv3_mis = cnn_mis.layers{4}.activations;

    % Correctly classified sample
    sample_cor = data_test(:,:,1,correct_idx);
    cnn_cor = cnn;
    cnn_cor = forward(cnn_cor, sample_cor, local_option);
    conv3_cor = cnn_cor.layers{4}.activations;

    % Select top filters (up to 16)
    conv3_channels = size(conv3_mis, 3);
    top_k = min(16, conv3_channels);
    conv3_mis_strength = squeeze(sum(sum(abs(conv3_mis), 1), 2));
    [~, top_idx_mis] = sort(conv3_mis_strength, 'descend');
    top_idx_mis = top_idx_mis(1:top_k);

    conv3_cor_strength = squeeze(sum(sum(abs(conv3_cor), 1), 2));
    [~, top_idx_cor] = sort(conv3_cor_strength, 'descend');
    top_idx_cor = top_idx_cor(1:top_k);

    % Save individual images for LaTeX subfigure layout

    % Misclassified input
    img = sample_mis;
    if invert_input_polarity
        img = 1 - img;
    end
    % Convert to uint8 and save at original 64x64 size
    img_uint8 = uint8(255 * img);
    imwrite(img_uint8, fullfile(output_dir, 'misclass_input.png'));

    % Misclassified Conv3 (top filters)
    mis_cols = ceil(sqrt(top_k));
    mis_rows = ceil(top_k / mis_cols);
    fig = figure('Position', [100, 100, 600, 600], 'Color', 'white');
    for i = 1:top_k
        subplot(mis_rows, mis_cols, i);
        filter_idx = top_idx_mis(i);
        imshow(conv3_mis(:,:,filter_idx), []);
        axis off;
    end
    set(gcf, 'Color', 'white');
    saveas(fig, fullfile(output_dir, 'misclass_conv3.png'));
    close(fig);

    % Correct input
    img = sample_cor;
    if invert_input_polarity
        img = 1 - img;
    end
    % Convert to uint8 and save at original 64x64 size
    img_uint8 = uint8(255 * img);
    imwrite(img_uint8, fullfile(output_dir, 'correct_input.png'));

    % Correct Conv3 (top filters)
    cor_cols = ceil(sqrt(top_k));
    cor_rows = ceil(top_k / cor_cols);
    fig = figure('Position', [100, 100, 600, 600], 'Color', 'white');
    for i = 1:top_k
        subplot(cor_rows, cor_cols, i);
        filter_idx = top_idx_cor(i);
        imshow(conv3_cor(:,:,filter_idx), []);
        axis off;
    end
    set(gcf, 'Color', 'white');
    saveas(fig, fullfile(output_dir, 'correct_conv3.png'));
    close(fig);

    fprintf('Saved: misclass_input.png, misclass_conv3.png, correct_input.png, correct_conv3.png\n');
end

%% Summary
fprintf('\n=== Visualization Complete ===\n');
fprintf('All figures saved to:\n');
fprintf('  %s\n\n', output_dir);
fprintf('Generated figures:\n');
fprintf('  1. class_samples.png - Dataset visualization\n');
fprintf('  2. training_curves.png - Training dynamics\n');
fprintf('  3. confusion_matrix.png - Classification confusion matrix\n');
fprintf('  4. correct_predictions.png - Correctly classified samples\n');
fprintf('  5. misclassifications.png - Misclassification examples\n');
fprintf('  6. feature_input.png - Input image for feature visualization\n');
fprintf('  7. feature_conv1.png - Conv1 feature maps (16 filters)\n');
fprintf('  8. feature_conv2.png - Conv2 feature maps (32 filters)\n');
fprintf('  9. feature_conv3.png - Conv3 feature maps (64 filters)\n');
fprintf('  10. misclass_input.png - Misclassified input\n');
fprintf('  11. misclass_conv3.png - Misclassified Conv3 features\n');
fprintf('  12. correct_input.png - Correctly classified input\n');
fprintf('  13. correct_conv3.png - Correctly classified Conv3 features\n');
fprintf('\nCopy desired figures to the separate report repository manually if needed.\n');
