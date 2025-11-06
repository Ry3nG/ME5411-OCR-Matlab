% task8_1_hyper.m
% Task 8: Part II - CNN Hyperparameter Sensitivity Analysis
%
% Tests CNN performance with different hyperparameters:
%   Group 1: Dropout rate (0.0, 0.1, 0.2, 0.3, 0.4)
%   Group 2: Learning rate (0.05, 0.1, 0.2)
%   Group 3: Batch size (64, 128, 256)
%
% For each hyperparameter, other parameters are fixed at baseline values.
% All experiments use baseline data (no augmentation).
%
% Metrics:
%   - Training accuracy
%   - Validation accuracy (synthetic test set)
%   - Real-world test accuracy (7M2-HD44780A00, 11 valid chars)
%   - Train-val gap (overfitting indicator)
%   - Val-to-realworld gap (distribution shift indicator)
%
% Usage:
%   matlab -batch "run('src/task8_1_hyper.m')"

clear all; %#ok<CLALL>
close all;

fprintf('\n');
fprintf('=========================================\n');
fprintf(' Task 8: CNN Hyperparameter Sensitivity \n');
fprintf('=========================================\n\n');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Set random seed for reproducibility (matches Task 7.1)
rng(0, 'twister');

%% Output directory
output_dir = fullfile(project_root, 'output', 'task8_1_hyper');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load datasets
fprintf('Loading datasets...\n');
train_data = load('data/train.mat');
test_data = load('data/test.mat');

data_train = train_data.data_train;
labels_train = train_data.labels_train + 1;  % Convert to 1-indexed
data_test = test_data.data_test;
labels_test = test_data.labels_test + 1;  % Convert to 1-indexed

fprintf('  Training samples: %d\n', size(data_train, 4));
fprintf('  Validation samples: %d\n\n', size(data_test, 4));

%% Load real-world test data
fprintf('Loading real-world test data (7M2-HD44780A00)...\n');
realworld_test = load_realworld_test_data();
fprintf('  Real-world test samples: %d\n\n', length(realworld_test.labels));

%% Define hyperparameter experiments
% Baseline configuration (from Task 7.1)
baseline = struct();
baseline.epochs = 30;
baseline.batch_size = 128;
baseline.learning_rate = 0.1;
baseline.lr_schedule = 'linear';
baseline.dropout = 0.2;

% Group 1: Dropout sensitivity
group1_name = 'dropout';
group1_param_name = 'Dropout Rate';
group1_values = [0.0, 0.1, 0.2, 0.3, 0.4];
group1_configs = cell(length(group1_values), 1);
for i = 1:length(group1_values)
    cfg = baseline;
    cfg.dropout = group1_values(i);
    cfg.name = sprintf('dropout_%.1f', group1_values(i));
    group1_configs{i} = cfg;
end

% Group 2: Learning rate sensitivity
group2_name = 'learning_rate';
group2_param_name = 'Learning Rate';
group2_values = [0.05, 0.1, 0.2];
group2_configs = cell(length(group2_values), 1);
for i = 1:length(group2_values)
    cfg = baseline;
    cfg.learning_rate = group2_values(i);
    cfg.name = sprintf('lr_%.3f', group2_values(i));
    group2_configs{i} = cfg;
end

% Group 3: Batch size sensitivity
group3_name = 'batch_size';
group3_param_name = 'Batch Size';
group3_values = [64, 128, 256];
group3_configs = cell(length(group3_values), 1);
for i = 1:length(group3_values)
    cfg = baseline;
    cfg.batch_size = group3_values(i);
    cfg.name = sprintf('batch_%d', group3_values(i));
    group3_configs{i} = cfg;
end

% Combine all groups (store configs as cell arrays)
experiment_groups = cell(3, 1);

experiment_groups{1} = struct();
experiment_groups{1}.name = group1_name;
experiment_groups{1}.param_name = group1_param_name;
experiment_groups{1}.values = group1_values;
experiment_groups{1}.configs = group1_configs;

experiment_groups{2} = struct();
experiment_groups{2}.name = group2_name;
experiment_groups{2}.param_name = group2_param_name;
experiment_groups{2}.values = group2_values;
experiment_groups{2}.configs = group2_configs;

experiment_groups{3} = struct();
experiment_groups{3}.name = group3_name;
experiment_groups{3}.param_name = group3_param_name;
experiment_groups{3}.values = group3_values;
experiment_groups{3}.configs = group3_configs;

%% Run experiments
all_results = cell(length(experiment_groups), 1);

for group_idx = 1:length(experiment_groups)
    group = experiment_groups{group_idx};

    fprintf('\n');
    fprintf('=========================================\n');
    fprintf('Parameter Group %d/%d: %s\n', group_idx, length(experiment_groups), group.param_name);
    fprintf('=========================================\n\n');

    group_results = struct();
    group_results.param_values = group.values;
    num_configs = numel(group.configs);
    group_results.train_acc = zeros(num_configs, 1);
    group_results.val_acc = zeros(num_configs, 1);
    group_results.realworld_acc = zeros(num_configs, 1);
    group_results.train_val_gap = zeros(num_configs, 1);
    group_results.val_realworld_gap = zeros(num_configs, 1);
    group_results.train_time = zeros(num_configs, 1);

    for cfg_idx = 1:num_configs
        cfg = group.configs{cfg_idx};

        fprintf('-----------------------------------\n');
        fprintf('Config %d/%d: %s\n', cfg_idx, num_configs, cfg.name);
        fprintf('  Dropout: %.2f\n', cfg.dropout);
        fprintf('  Learning rate: %.3f\n', cfg.learning_rate);
        fprintf('  Batch size: %d\n', cfg.batch_size);
        fprintf('-----------------------------------\n');

        % Create directory for this config
        cfg_dir = fullfile(output_dir, group.name, cfg.name);
        if ~exist(cfg_dir, 'dir')
            mkdir(cfg_dir);
        end

        % Check if model already exists
        model_file = fullfile(cfg_dir, 'model.mat');
        if exist(model_file, 'file')
            fprintf('Loading existing model...\n');
            loaded = load(model_file);
            net = loaded.net;
            train_time = loaded.train_time;
            fprintf('  Loaded (trained in %.2f min)\n', train_time / 60);
        else
            % Train CNN
            fprintf('Training CNN...\n');
            tic;
            cfg.output_dir = cfg_dir;
            [net, history] = train_cnn(data_train, labels_train, ...
                                       data_test, labels_test, cfg);
            train_time = toc;
            fprintf('Training completed in %.2f min\n', train_time / 60);

            % Save model
            save(model_file, 'net', 'history', 'train_time', '-v7.3');
        end

        % Evaluate on training set
        fprintf('Evaluating on training set...\n');
        [~, train_acc] = evaluate_cnn(net, data_train, labels_train);
        fprintf('  Training accuracy: %.2f%%\n', train_acc * 100);

        % Evaluate on validation set
        fprintf('Evaluating on validation set...\n');
        [~, val_acc] = evaluate_cnn(net, data_test, labels_test);
        fprintf('  Validation accuracy: %.2f%%\n', val_acc * 100);

        % Evaluate on real-world test
        fprintf('Evaluating on real-world test...\n');
        [realworld_pred_all, ~] = evaluate_cnn(net, realworld_test.images, []);
        % Extract in-vocabulary predictions
        realworld_pred = realworld_pred_all(realworld_test.invocab_positions) + 1;  % Convert to 1-indexed
        realworld_correct = sum(realworld_pred == realworld_test.labels);
        realworld_acc = realworld_correct / length(realworld_test.labels);
        fprintf('  Real-world accuracy: %.2f%% (%d/%d)\n', ...
                realworld_acc * 100, realworld_correct, length(realworld_test.labels));

        % Calculate gaps
        train_val_gap = train_acc - val_acc;
        val_realworld_gap = val_acc - realworld_acc;

        fprintf('  Train-val gap: %.2f%%\n', train_val_gap * 100);
        fprintf('  Val-to-realworld gap: %.2f%%\n\n', val_realworld_gap * 100);

        % Store results
        group_results.train_acc(cfg_idx) = train_acc;
        group_results.val_acc(cfg_idx) = val_acc;
        group_results.realworld_acc(cfg_idx) = realworld_acc;
        group_results.train_val_gap(cfg_idx) = train_val_gap;
        group_results.val_realworld_gap(cfg_idx) = val_realworld_gap;
        group_results.train_time(cfg_idx) = train_time;
    end

    all_results{group_idx} = group_results;

    % Save group results
    save(fullfile(output_dir, group.name, 'results.mat'), 'group_results', 'group');

    % Generate group visualization
    generate_sensitivity_plots(group, group_results, ...
                              fullfile(output_dir, group.name));
end

%% Generate summary report
fprintf('\n');
fprintf('=========================================\n');
fprintf('Generating summary report...\n');
fprintf('=========================================\n\n');

generate_hyper_summary_report(experiment_groups, all_results, output_dir);

fprintf('All experiments completed!\n');
fprintf('Results saved to: %s\n\n', output_dir);


%% Helper Functions

function [net, history] = train_cnn(data_train, labels_train, data_test, labels_test, cfg)
    % Train CNN with given configuration

    % Define CNN architecture (same as Task 7.1)
    cnn_arch = struct();
    cnn_arch.layers = {
        struct('type', 'input')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4,  'poolDim', 4, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 4, 'numFilters', 8,  'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', cfg.dropout)
        struct('type', 'Linear', 'hiddenUnits', 50,  'actiFunc', 'relu')
        struct('type', 'output', 'softmax', 1)
    };

    % Training options
    options = struct();
    options.epochs = cfg.epochs;
    options.minibatch = cfg.batch_size;
    options.lr_max = cfg.learning_rate;
    options.lr = cfg.learning_rate;
    options.lr_min = 1e-5;
    options.lr_method = cfg.lr_schedule;
    options.lr_duty = 20;
    options.momentum = 0.9;
    if isfield(cfg, 'output_dir') && ~isempty(cfg.output_dir)
        log_dir = cfg.output_dir;
    else
        log_dir = fullfile(pwd, 'temp_logs');
    end
    if ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end
    log_dir = string(log_dir);
    if ~endsWith(log_dir, string(filesep))
        log_dir = log_dir + string(filesep);
    end
    options.log_path = log_dir;
    options.l2_penalty = 0.01;
    options.use_l2 = false;
    options.save_best_acc_model = true;
    options.train_mode = true;

    num_train_samples = size(data_train, 4);
    iter_per_epoch = max(1, floor((num_train_samples - options.minibatch) / options.minibatch) + 1);
    options.total_iter = iter_per_epoch * options.epochs;

    % Initialize network
    numClasses = 7;
    net = initModelParams(cnn_arch, data_train, numClasses);

    % Train
    net = learn(net, data_train, labels_train, data_test, labels_test, options);

    % Extract history from net if available
    if isfield(net, 'history')
        history = net.history;
    else
        history = struct();
    end
end


function [predictions, accuracy] = evaluate_cnn(net, data, labels)
    % Evaluate CNN on given data

    % Handle cell array (from real-world test data)
    if iscell(data)
        % Convert cell array to 4D array
        num_images = length(data);
        data_array = zeros(64, 64, 1, num_images);
        for i = 1:num_images
            data_array(:, :, 1, i) = data{i};
        end
        % Apply polarity correction for real-world data
        data_array = 1 - data_array;
        data = data_array;
    end

    [preds_raw, ~] = predict(net, data);
    predictions = preds_raw - 1;  % Convert back to 0-indexed

    if isempty(labels)
        accuracy = NaN;
    else
        labels_eval = labels - 1;  % Convert to 0-indexed
        accuracy = sum(predictions == labels_eval) / length(predictions);
    end
end


function realworld_data = load_realworld_test_data()
    % Load Task 7.3 real-world test data (both upper and lower parts)
    % Reuses Task 7.3 segmentation code with custom implementations

    % Upper part: ZM2 (3 chars, only '7' in vocabulary)
    % Lower part: HD44780A00 (10 chars, all in vocabulary)
    % Total: 13 chars, 11 in-vocabulary

    % Process Upper Part (ZM2)
    upper_img_path = 'data/cropped_charact2/cropped_lower.png';
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
        [h, w] = size(char_img);
        max_dim = max(h, w);
        padded = zeros(max_dim, max_dim);
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
    invocab_positions = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    ground_truth_0indexed = [2, 6, 5, 1, 1, 2, 3, 0, 4, 0, 0];

    % Package data
    realworld_data = struct();
    realworld_data.images = char_images;
    realworld_data.num_total = length(all_chars);
    realworld_data.invocab_positions = invocab_positions;
    realworld_data.labels = ground_truth_0indexed' + 1;  % Convert to 1-indexed
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


function generate_sensitivity_plots(group, results, output_dir)
    % Generate sensitivity analysis plots

    param_values = results.param_values;

    % Plot 1: Accuracy comparison
    figure('Position', [100, 100, 1200, 400]);

    subplot(1, 3, 1);
    plot(param_values, results.train_acc * 100, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot(param_values, results.val_acc * 100, 's-', 'LineWidth', 2, 'MarkerSize', 8);
    plot(param_values, results.realworld_acc * 100, '^-', 'LineWidth', 2, 'MarkerSize', 8);
    hold off;
    xlabel(group.param_name);
    ylabel('Accuracy (%)');
    title('Accuracy vs Parameter');
    legend('Training', 'Validation', 'Real-world', 'Location', 'best');
    grid on;

    subplot(1, 3, 2);
    plot(param_values, results.train_val_gap * 100, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel(group.param_name);
    ylabel('Gap (%)');
    title('Train-Val Gap (Overfitting Indicator)');
    grid on;

    subplot(1, 3, 3);
    plot(param_values, results.val_realworld_gap * 100, 's-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel(group.param_name);
    ylabel('Gap (%)');
    title('Val-to-Realworld Gap (Distribution Shift)');
    grid on;

    saveas(gcf, fullfile(output_dir, 'sensitivity_curves.png'));
    close(gcf);

    fprintf('  Saved sensitivity curves to: %s\n', fullfile(output_dir, 'sensitivity_curves.png'));
end


function generate_hyper_summary_report(experiment_groups, all_results, output_dir)
    % Generate summary report comparing all hyperparameters

    % Create summary table
    fid = fopen(fullfile(output_dir, 'summary.txt'), 'w');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'CNN Hyperparameter Sensitivity Summary\n');
    fprintf(fid, '========================================\n\n');

    for group_idx = 1:length(experiment_groups)
        group = experiment_groups{group_idx};
        results = all_results{group_idx};

        fprintf(fid, 'Parameter: %s\n', group.param_name);
        fprintf(fid, '-----------------------------------\n');
        fprintf(fid, 'Value\tTrain\tVal\tReal\tTr-Val Gap\tVal-Real Gap\n');

        for i = 1:length(results.param_values)
            fprintf(fid, '%.3f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n', ...
                    results.param_values(i), ...
                    results.train_acc(i) * 100, ...
                    results.val_acc(i) * 100, ...
                    results.realworld_acc(i) * 100, ...
                    results.train_val_gap(i) * 100, ...
                    results.val_realworld_gap(i) * 100);
        end
        fprintf(fid, '\n');
    end

    fclose(fid);
    fprintf('Summary report saved to: %s\n', fullfile(output_dir, 'summary.txt'));
end
