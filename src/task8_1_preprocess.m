% task8_1_preprocess.m
% Task 8.1: CNN Data Preprocessing Sensitivity Analysis
%
% Tests CNN performance with different data augmentation strategies:
%   1. Baseline (no augmentation)
%   2. Noise augmentation
%   3. Scale augmentation
%   4. Rotation augmentation
%   5. Combined augmentation
%
% Outputs:
%   - 5 trained CNN models
%   - Performance comparison table
%   - Training curves
%   - Real-world test results (7M2+HD44780A00)
%   - Ablation analysis figure
%
% Usage:
%   matlab -batch "run('src/task8_1_preprocess.m')"

clear all; %#ok<CLALL>
close all;

fprintf('\n');
fprintf('=========================================\n');
fprintf('  Task 8.1: CNN Preprocessing Analysis  \n');
fprintf('=========================================\n\n');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Set random seed (MUST match Task 7.1 for reproducibility)
rng(0, 'twister');

%% Configuration
output_dir = fullfile(project_root, 'output', 'task8_1_preprocess');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% CNN hyperparameters (from Task 7.1 baseline)
cnn_params = struct();
cnn_params.img_size = 64;
cnn_params.num_classes = 7;
cnn_params.epochs = 30;
cnn_params.batch_size = 128;
cnn_params.learning_rate = 0.1;
cnn_params.lr_schedule = 'linear';
cnn_params.dropout_rate = 0.2;

fprintf('CNN Configuration:\n');
fprintf('  Image size: %dx%d\n', cnn_params.img_size, cnn_params.img_size);
fprintf('  Epochs: %d\n', cnn_params.epochs);
fprintf('  Batch size: %d\n', cnn_params.batch_size);
fprintf('  Learning rate: %.3f\n', cnn_params.learning_rate);
fprintf('  Dropout: %.2f\n\n', cnn_params.dropout_rate);

%% Define augmentation strategies
strategies = {
    'baseline',   'data/train.mat',           'Baseline (no augmentation)'
    'noise',      'data/train_noise.mat',     'Gaussian noise (σ=0.05) + Salt-and-pepper (2%)'
    'scale',      'data/train_scale.mat',     'Scale (0.8-1.2x)'
    'rotation',   'data/train_rotation.mat',  'Rotation (±20°)'
    'combined',   'data/train_combined.mat',  'Combined (all augmentations)'
};

fprintf('Augmentation Strategies:\n');
for i = 1:size(strategies, 1)
    fprintf('  %d. %-10s: %s\n', i, strategies{i, 1}, strategies{i, 3});
end
fprintf('\n');

%% Load test set (same for all experiments)
fprintf('Loading test set...\n');
test_data = load('data/test.mat');
data_test = test_data.data_test;
labels_test = test_data.labels_test;
% Convert labels to 1-indexed for MATLAB operations (0-indexed -> 1-indexed)
labels_test = labels_test + 1;
fprintf('  Test samples: %d\n\n', size(data_test, 4));

%% Load real-world test data (Task 7.3)
fprintf('Loading real-world test data (HD44780A00)...\n');
realworld_test = load_realworld_test_data();
fprintf('  Real-world test samples: %d\n\n', length(realworld_test.labels));

%% Train and evaluate for each strategy
results = struct();
training_histories = cell(size(strategies, 1), 1);

for i = 1:size(strategies, 1)
    strategy_name = strategies{i, 1};
    data_file = strategies{i, 2};
    strategy_desc = strategies{i, 3};

    fprintf('=========================================\n');
    fprintf('Strategy %d/%d: %s\n', i, size(strategies, 1), strategy_name);
    fprintf('Description: %s\n', strategy_desc);
    fprintf('=========================================\n');

    % Create subfolder for this strategy
    strategy_dir = fullfile(output_dir, strategy_name);
    if ~exist(strategy_dir, 'dir')
        mkdir(strategy_dir);
    end

    % Check if model already exists
    model_file = fullfile(strategy_dir, 'model.mat');
    if exist(model_file, 'file')
        fprintf('Loading existing model from: %s\n', model_file);
        loaded = load(model_file);
        net = loaded.net;
        history = loaded.history;
        train_time = loaded.train_time;
        training_histories{i} = history;
        fprintf('  Model loaded (trained in %.2f minutes)\n', train_time / 60);
    else
        % Load training data
        fprintf('Loading training data: %s\n', data_file);
        train_data = load(data_file);
        data_train = train_data.data_train;
        labels_train = train_data.labels_train;
        % Convert labels to 1-indexed for MATLAB operations (0-indexed -> 1-indexed)
        labels_train = labels_train + 1;
        fprintf('  Training samples: %d\n', size(data_train, 4));

        % Train CNN
        fprintf('Training CNN...\n');
        tic;
        [net, history] = train_cnn(data_train, labels_train, data_test, labels_test, cnn_params);
        train_time = toc;

        training_histories{i} = history;

        fprintf('Training completed in %.2f minutes\n', train_time / 60);

        % Save model immediately
        fprintf('Saving model to: %s\n', model_file);
        save(model_file, 'net', 'history', 'train_time', '-v7.3');
    end

    % Evaluate on validation set
    fprintf('Evaluating on validation set...\n');
    [val_pred, val_accuracy] = evaluate_cnn(net, data_test, labels_test);

    % Evaluate on real-world test set
    fprintf('Evaluating on real-world test set (HD44780A00)...\n');
    [real_pred, real_accuracy, real_correct] = evaluate_realworld(net, realworld_test);

    % Store results
    results.(strategy_name) = struct();
    results.(strategy_name).net = net;
    results.(strategy_name).val_accuracy = val_accuracy;
    results.(strategy_name).real_accuracy = real_accuracy;
    results.(strategy_name).real_correct = real_correct;
    results.(strategy_name).real_total = length(realworld_test.labels);
    results.(strategy_name).train_time = train_time;
    results.(strategy_name).history = history;

    fprintf('Results:\n');
    fprintf('  Validation accuracy: %.2f%%\n', val_accuracy * 100);
    fprintf('  Real-world accuracy: %.2f%% (%d/%d)\n', real_accuracy * 100, ...
        real_correct, length(realworld_test.labels));
    fprintf('  Training time: %.2f minutes\n\n', train_time / 60);
end

%% Save all results
fprintf('Saving results...\n');
save(fullfile(output_dir, 'results.mat'), 'results', 'training_histories', 'strategies', 'cnn_params');

%% Generate visualizations
fprintf('\n=========================================\n');
fprintf('Generating visualizations...\n');
fprintf('=========================================\n');

% 1. Training curves comparison
plot_training_curves(training_histories, strategies, output_dir);

% 2. Performance comparison table
generate_comparison_table(results, strategies, output_dir);

% 3. Ablation analysis (bar chart)
plot_ablation_analysis(results, strategies, output_dir);

fprintf('\n=========================================\n');
fprintf('✓ Task 8.1 completed successfully!\n');
fprintf('=========================================\n');
fprintf('\nResults saved to: %s\n', output_dir);
fprintf('\nSummary:\n');
for i = 1:size(strategies, 1)
    strategy_name = strategies{i, 1};
    res = results.(strategy_name);
    fprintf('  %-10s: Val=%.2f%%, Real=%d/%d (%.1f%%), Time=%.1fmin\n', ...
        strategy_name, res.val_accuracy*100, res.real_correct, res.real_total, ...
        res.real_accuracy*100, res.train_time/60);
end
fprintf('\n');

%% Helper Functions

function [net, history] = train_cnn(data_train, labels_train, data_test, labels_test, params)
    % Train CNN using Task 7.1 architecture

    % Initialize network
    net = struct();
    net.img_size = params.img_size;
    net.num_classes = params.num_classes;

    % Architecture (LeNet-like, compatible with initModelParams)
    % Format: input -> Conv2D (with pooling) -> Conv2D (with pooling) -> Linear -> Linear -> output
    net.layers = {
        struct('type', 'input')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 6, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Linear', 'hiddenUnits', 120, 'actiFunc', 'relu')
        struct('type', 'Linear', 'hiddenUnits', 84, 'actiFunc', 'relu', 'dropout', params.dropout_rate)
        struct('type', 'output', 'softmax', 1)
    };

    % Initialize parameters using initModelParams
    numClasses = params.num_classes;
    net = initModelParams(net, data_train, numClasses);

    % Create temporary log directory for training history
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    temp_log_dir = fullfile(project_root, 'output', 'task8_1_preprocess', 'temp_logs');
    if ~exist(temp_log_dir, 'dir')
        mkdir(temp_log_dir);
    end

    % Training parameters (compatible with learn function)
    train_params = struct();
    train_params.epochs = params.epochs;
    train_params.minibatch = params.batch_size;
    train_params.lr_max = params.learning_rate;
    train_params.lr = params.learning_rate;
    train_params.lr_min = 1e-5;
    train_params.lr_method = params.lr_schedule;
    train_params.lr_duty = 20;
    train_params.momentum = 0.9;
    train_params.l2_penalty = 0.0005;
    train_params.use_l2 = true;
    train_params.verbose = true;
    train_params.save_best_acc_model = false;
    train_params.train_mode = true;
    train_params.log_path = string(temp_log_dir) + "/";  % Convert to string for + operator

    % Set learning rate schedule parameters if needed
    if strcmp(params.lr_schedule, 'step')
        train_params.lr_step_size = 10;
        train_params.lr_gamma = 0.1;
    elseif strcmp(params.lr_schedule, 'exp')
        train_params.lr_decay_rate = 0.95;
    end

    % Train
    net = learn(net, data_train, labels_train, data_test, labels_test, train_params);

    % Load training history from saved files
    try
        acc_test_data = load(fullfile(temp_log_dir, 'acc_test.mat'));
        acc_train_data = load(fullfile(temp_log_dir, 'acc_train.mat'));
        history = struct();
        history.val_accuracy = acc_test_data.acc_test;
        history.train_accuracy = acc_train_data.acc_train;
    catch
        % If files don't exist, create empty history
        history = struct();
        history.val_accuracy = [];
        history.train_accuracy = [];
    end
end

function [predictions, accuracy] = evaluate_cnn(net, data, labels)
    % Evaluate CNN on test data
    % Directly predict on entire dataset (like Task 7.1)

    [predictions, ~] = predict(net, data);
    accuracy = sum(predictions == labels) / length(labels);
end

function realworld_data = load_realworld_test_data()
    % Load Task 7.3 real-world test data (both upper and lower parts)
    % Reuses Task 7.3 segmentation code with custom implementations

    % Upper part: ZM2 (3 chars, only '7' in vocabulary)
    % Lower part: HD44780A00 (10 chars, all in vocabulary)
    % Total: 13 chars, 11 in-vocabulary

    % Process Upper Part (ZM2)
    upper_img_path = 'data/cropped_charact2/cropped_lower.png';  % Actually contains ZM2
    img_upper = imread(upper_img_path);
    if size(img_upper, 3) == 3
        img_upper = myRgb2gray(img_upper);
    end
    threshold_upper = myOtsuThres(img_upper);
    binary_upper = myImbinarize(img_upper, threshold_upper);

    % Segment upper characters (with merge detection, like Task 7.3)
    [chars_upper, ~] = segment_characters(binary_upper, 800);

    % Process Lower Part (HD44780A00)
    lower_img_path = 'data/cropped_charact2/cropped_HD44780A00.png';
    img_lower = imread(lower_img_path);
    if size(img_lower, 3) == 3
        img_lower = myRgb2gray(img_lower);
    end
    threshold_lower = myOtsuThres(img_lower);
    binary_lower = myImbinarize(img_lower, threshold_lower);

    % Segment lower characters (with merge detection, like Task 7.3)
    [chars_lower, ~] = segment_characters(binary_lower, 300);

    % Combine all characters (upper + lower)
    all_chars = [chars_upper, chars_lower];

    % Prepare all characters for classification
    char_images = cell(length(all_chars), 1);
    for i = 1:length(all_chars)
        char_img = all_chars{i};

        % Pad to square (use zeros/black like Task 7.3)
        % This preserves the original background characteristics after polarity inversion
        [h, w] = size(char_img);
        max_dim = max(h, w);
        padded = zeros(max_dim, max_dim);  % BLACK padding (like Task 7.3:147)
        y_start = floor((max_dim - h) / 2) + 1;
        x_start = floor((max_dim - w) / 2) + 1;
        padded(y_start:y_start+h-1, x_start:x_start+w-1) = double(char_img);

        % Resize to 64x64
        resized = myImresize(padded, [64, 64], 'bilinear');
        char_images{i} = resized;
    end

    % In-vocabulary positions (1-indexed): [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    % Ground truth for in-vocabulary chars: 7, H, D, 4, 4, 7, 8, 0, A, 0, 0
    % Converted to 0-indexed class indices: [2, 6, 5, 1, 1, 2, 3, 0, 4, 0, 0]
    % Need to convert to 1-indexed for MATLAB: add 1
    invocab_positions = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    ground_truth_0indexed = [2, 6, 5, 1, 1, 2, 3, 0, 4, 0, 0];  % 7, H, D, 4, 4, 7, 8, 0, A, 0, 0

    % Package data
    realworld_data = struct();
    realworld_data.images = char_images;
    realworld_data.num_total = length(all_chars);  % 13
    realworld_data.invocab_positions = invocab_positions;
    realworld_data.labels = ground_truth_0indexed' + 1;  % Convert to 1-indexed
    realworld_data.expected_str = '7 (from ZM2) + HD44780A00';
end

function [chars, bboxes] = segment_characters(binary_img, min_area)
    % Character segmentation with merged character detection (from Task 7.3)

    CC = myBwconncomp(binary_img);
    props = myRegionprops(CC, 'BoundingBox', 'Area');

    % Filter by area
    valid_idx = [];
    for i = 1:length(props)
        if props(i).Area >= min_area
            valid_idx = [valid_idx, i]; %#ok<AGROW>
        end
    end

    % Sort by x-coordinate
    bboxes_all = zeros(length(valid_idx), 4);
    for i = 1:length(valid_idx)
        bboxes_all(i, :) = props(valid_idx(i)).BoundingBox;
    end
    [~, sort_idx] = sort(bboxes_all(:, 1));
    sorted_bboxes = bboxes_all(sort_idx, :);

    % Segment characters (split merged characters if needed)
    chars = {};
    bboxes = [];

    min_width = min(sorted_bboxes(:, 3));
    merge_threshold = min_width * 1.8;

    for i = 1:size(sorted_bboxes, 1)
        bbox = sorted_bboxes(i, :);
        x = round(bbox(1));
        y = round(bbox(2));
        w = round(bbox(3));
        h = round(bbox(4));

        char_region = binary_img(y:y+h-1, x:x+w-1);

        % Check if merged character
        if w > merge_threshold
            split_point = round(w / 2);

            char1 = char_region(:, 1:split_point);
            char2 = char_region(:, split_point+1:end);

            chars{end+1} = char1; %#ok<AGROW>
            bboxes = [bboxes; x, y, split_point, h]; %#ok<AGROW>

            chars{end+1} = char2; %#ok<AGROW>
            bboxes = [bboxes; x+split_point, y, w-split_point, h]; %#ok<AGROW>
        else
            chars{end+1} = char_region; %#ok<AGROW>
            bboxes = [bboxes; bbox]; %#ok<AGROW>
        end
    end
end

function [predictions, accuracy, num_correct] = evaluate_realworld(net, realworld_data)
    % Evaluate on real-world test data
    % Predict all chars but only evaluate in-vocabulary ones

    % Convert cell array to 4D array (64x64x1xN)
    data_array = zeros(64, 64, 1, realworld_data.num_total);
    for i = 1:realworld_data.num_total
        data_array(:, :, 1, i) = realworld_data.images{i};
    end

    % Apply polarity correction (like Task 7.3)
    % Real-world images have different polarity than training data
    data_array = 1 - data_array;

    % Predict all at once (like Task 7.3)
    [all_predictions, ~] = predict(net, data_array);

    % Extract in-vocabulary predictions
    predictions = all_predictions(realworld_data.invocab_positions);

    % Compare with ground truth
    num_correct = sum(predictions == realworld_data.labels);
    accuracy = num_correct / length(realworld_data.labels);
end

function plot_training_curves(histories, strategies, output_dir)
    % Plot training and validation accuracy curves

    figure('Position', [100, 100, 1200, 400], 'Color', 'w');

    colors = lines(length(histories));

    for i = 1:length(histories)
        history = histories{i};
        strategy_name = strategies{i, 1};

        plot(1:length(history.val_accuracy), history.val_accuracy * 100, ...
            'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', strategy_name);
        hold on;
    end

    xlabel('Epoch', 'FontSize', 12);
    ylabel('Validation Accuracy (%)', 'FontSize', 12);
    title('CNN Training Curves: Data Augmentation Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'southeast', 'FontSize', 10);
    set(gca, 'Color', 'w');  % Set axes background to white
    grid on;

    output_file = fullfile(output_dir, 'training_curves.png');
    % Export with white background
    exportgraphics(gcf, output_file, 'BackgroundColor', 'white', 'Resolution', 300);
    fprintf('  Saved: training_curves.png\n');
    close(gcf);
end

function generate_comparison_table(results, strategies, output_dir)
    % Generate and save comparison table

    % Create table
    strategy_names = strategies(:, 1);
    val_acc = zeros(length(strategy_names), 1);
    real_acc = zeros(length(strategy_names), 1);
    real_str = cell(length(strategy_names), 1);
    train_time = zeros(length(strategy_names), 1);

    for i = 1:length(strategy_names)
        name = strategy_names{i};
        res = results.(name);
        val_acc(i) = res.val_accuracy * 100;
        real_acc(i) = res.real_accuracy * 100;
        real_str{i} = sprintf('%d/%d', res.real_correct, res.real_total);
        train_time(i) = res.train_time / 60;
    end

    % Save as text
    fid = fopen(fullfile(output_dir, 'comparison_table.txt'), 'w');
    fprintf(fid, 'CNN Data Augmentation Sensitivity Analysis\n');
    fprintf(fid, '==========================================\n\n');
    fprintf(fid, '%-12s | Val Acc (%%) | Real Test    | Real Acc (%%) | Time (min)\n', 'Strategy');
    fprintf(fid, '-------------|-------------|--------------|--------------|------------\n');
    for i = 1:length(strategy_names)
        fprintf(fid, '%-12s | %10.2f | %12s | %11.1f | %9.1f\n', ...
            strategy_names{i}, val_acc(i), real_str{i}, real_acc(i), train_time(i));
    end
    fclose(fid);

    fprintf('  Saved: comparison_table.txt\n');
end

function plot_ablation_analysis(results, strategies, output_dir)
    % Plot ablation analysis (bar chart)

    figure('Position', [100, 100, 1000, 500], 'Color', 'w');

    strategy_names = strategies(:, 1);
    val_acc = zeros(length(strategy_names), 1);
    real_acc = zeros(length(strategy_names), 1);

    for i = 1:length(strategy_names)
        name = strategy_names{i};
        res = results.(name);
        val_acc(i) = res.val_accuracy * 100;
        real_acc(i) = res.real_accuracy * 100;
    end

    % Create grouped bar chart
    x = 1:length(strategy_names);
    width = 0.35;

    bar(x - width/2, val_acc, width, 'FaceColor', [0.2, 0.6, 0.8]);
    hold on;
    bar(x + width/2, real_acc, width, 'FaceColor', [0.8, 0.4, 0.2]);

    % Add value labels
    for i = 1:length(strategy_names)
        text(x(i) - width/2, val_acc(i) + 1, sprintf('%.1f', val_acc(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        text(x(i) + width/2, real_acc(i) + 1, sprintf('%.1f', real_acc(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end

    set(gca, 'XTick', x);
    set(gca, 'XTickLabel', strategy_names);
    set(gca, 'Color', 'w');  % Set axes background to white
    xlabel('Augmentation Strategy', 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    title('CNN Ablation Analysis: Augmentation Impact', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'Validation Set', 'Real-world Test (HD44780A00)'}, 'Location', 'southeast', 'FontSize', 10);
    grid on;
    ylim([0, 105]);

    output_file = fullfile(output_dir, 'ablation_analysis.png');
    % Export with white background
    exportgraphics(gcf, output_file, 'BackgroundColor', 'white', 'Resolution', 300);
    fprintf('  Saved: ablation_analysis.png\n');
    close(gcf);
end
