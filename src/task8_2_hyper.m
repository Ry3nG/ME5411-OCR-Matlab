% task8_2_hyper.m
% Task 8: Part II - SVM Hyperparameter Sensitivity Analysis
%
% Tests SOM+BoW+SVM pipeline performance with different hyperparameters:
%   Group 1: SOM Codebook Size (50, 100, 150, 200)
%   Group 2: Spatial Pyramid (1x1, 1x1+2x2, 1x1+2x2+3x3)
%   Group 3: Soft Voting Sigma (0.5, 0.75, 1.0, 1.5)
%
% For each hyperparameter, other parameters are fixed at baseline values.
% All experiments use baseline data (no augmentation).
%
% Metrics:
%   - Training accuracy (training set)
%   - Validation accuracy (synthetic test set)
%   - Real-world test accuracy (7M2-HD44780A00, 11 valid chars)
%   - Train-to-val gap (overfitting indicator)
%   - Feature dimensionality (for spatial pyramid analysis)
%
% Usage:
%   matlab -batch "run('src/task8_2_hyper.m')"

clear all; %#ok<CLALL>
close all;

fprintf('\n');
fprintf('=========================================\n');
fprintf(' Task 8: SVM Hyperparameter Sensitivity \n');
fprintf('=========================================\n\n');

%% Setup
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));
addpath(genpath('src/viz'));

% Set random seed for reproducibility (matches Task 7.2)
rng(0, 'twister');

%% Output directory
output_dir = fullfile(project_root, 'output', 'task8_2_hyper');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load datasets
fprintf('Loading datasets...\n');
train_data = load('data/train.mat');
test_data = load('data/test.mat');

Xtr = train_data.data_train;
Ytr = train_data.labels_train + 1;  % Convert to 1-indexed
Xte = test_data.data_test;
Yte = test_data.labels_test + 1;  % Convert to 1-indexed

[H, W, ~, N_train] = size(Xtr);
N_test = size(Xte, 4);
num_classes = 7;

fprintf('  Training samples: %d\n', N_train);
fprintf('  Validation samples: %d\n\n', N_test);

%% Load real-world test data
fprintf('Loading real-world test data (7M2-HD44780A00)...\n');
realworld_test = load_realworld_test_data();
fprintf('  Real-world test samples: %d\n\n', length(realworld_test.labels));

%% Define hyperparameter experiments
% Baseline configuration (from Task 7.2)
baseline = struct();
baseline.patch_size = 8;
baseline.num_patch_samples = 100000;
baseline.som_grid = [10, 10];  % Codebook size = 100
baseline.som_iterations = 50000;
baseline.som_lr_init = 0.5;
baseline.som_lr_final = 0.01;
baseline.som_sigma_init = 5.0;
baseline.som_sigma_final = 0.5;
baseline.som_batch = 32;
baseline.stride = 4;
baseline.soft_voting = true;
baseline.sigma_bow = 0.75;
baseline.spatial_pyramid = true;
baseline.pyramid_levels = [1, 2];  % 1x1 + 2x2
baseline.C = 1.0;
baseline.svm_epochs = 200;
baseline.svm_lr = 0.01;
baseline.use_pca = true;
baseline.pca_var = 0.95;
baseline.min_patch_std = 0.001;

% Group 1: SOM Codebook Size sensitivity
group1_name = 'codebook_size';
group1_param_name = 'SOM Codebook Size';
group1_values = [50, 100, 150, 200];
group1_configs = cell(length(group1_values), 1);
for i = 1:length(group1_values)
    cfg = baseline;
    grid_size = round(sqrt(group1_values(i)));
    cfg.som_grid = [grid_size, grid_size];
    cfg.name = sprintf('codebook_%d', group1_values(i));
    group1_configs{i} = cfg;
end

% Group 2: Spatial Pyramid sensitivity
group2_name = 'spatial_pyramid';
group2_param_name = 'Spatial Pyramid';
group2_labels = {'1x1', '1x1+2x2', '1x1+2x2+3x3'};
group2_configs = cell(3, 1);
% Config 1: 1x1 only (100-D)
cfg = baseline;
cfg.pyramid_levels = [1];
cfg.name = 'pyramid_1x1';
group2_configs{1} = cfg;
% Config 2: 1x1 + 2x2 (500-D, baseline)
cfg = baseline;
cfg.pyramid_levels = [1, 2];
cfg.name = 'pyramid_1x1_2x2';
group2_configs{2} = cfg;
% Config 3: 1x1 + 2x2 + 3x3 (1000-D)
cfg = baseline;
cfg.pyramid_levels = [1, 2, 3];
cfg.name = 'pyramid_1x1_2x2_3x3';
group2_configs{3} = cfg;

% Group 3: Soft Voting Sigma sensitivity
group3_name = 'soft_voting_sigma';
group3_param_name = 'Soft Voting Sigma';
group3_values = [0.5, 0.75, 1.0, 1.5];
group3_configs = cell(length(group3_values), 1);
for i = 1:length(group3_values)
    cfg = baseline;
    cfg.sigma_bow = group3_values(i);
    cfg.name = sprintf('sigma_%.2f', group3_values(i));
    group3_configs{i} = cfg;
end

% Combine all groups (store configs as cell arrays instead of struct arrays)
experiment_groups = cell(3, 1);

experiment_groups{1} = struct();
experiment_groups{1}.name = group1_name;
experiment_groups{1}.param_name = group1_param_name;
experiment_groups{1}.values = group1_values;
experiment_groups{1}.configs = group1_configs;
experiment_groups{1}.value_labels = arrayfun(@num2str, group1_values, 'UniformOutput', false);

experiment_groups{2} = struct();
experiment_groups{2}.name = group2_name;
experiment_groups{2}.param_name = group2_param_name;
experiment_groups{2}.values = 1:length(group2_labels);
experiment_groups{2}.configs = group2_configs;
experiment_groups{2}.value_labels = group2_labels;

experiment_groups{3} = struct();
experiment_groups{3}.name = group3_name;
experiment_groups{3}.param_name = group3_param_name;
experiment_groups{3}.values = group3_values;
experiment_groups{3}.configs = group3_configs;
experiment_groups{3}.value_labels = arrayfun(@num2str, group3_values, 'UniformOutput', false);

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
    group_results.feature_dim = zeros(num_configs, 1);
    group_results.train_time = zeros(num_configs, 1);

    for cfg_idx = 1:num_configs
        cfg = group.configs{cfg_idx};

        fprintf('-----------------------------------\n');
        fprintf('Config %d/%d: %s\n', cfg_idx, num_configs, cfg.name);
        fprintf('  SOM grid: %dx%d (codebook=%d)\n', ...
                cfg.som_grid(1), cfg.som_grid(2), prod(cfg.som_grid));
        fprintf('  Spatial pyramid: %s\n', mat2str(cfg.pyramid_levels));
        fprintf('  Soft voting sigma: %.2f\n', cfg.sigma_bow);
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
            som = loaded.som;
            models = loaded.models;
            pca_model = loaded.pca_model;
            train_time = loaded.train_time;
            feature_dim = loaded.feature_dim;
            fprintf('  Loaded (trained in %.2f min)\n', train_time / 60);
        else
            % Train SOM + SVM
            fprintf('Training SOM + SVM pipeline...\n');
            train_start = tic;
            [som, models, pca_model, feature_dim] = ...
                train_svm_pipeline(Xtr, Ytr, cfg);
            train_time = toc(train_start);
            fprintf('Training completed in %.2f min\n', train_time / 60);

            % Save model
            save(model_file, 'som', 'models', 'pca_model', ...
                 'train_time', 'feature_dim', '-v7.3');
        end

        % Evaluate on training set
        fprintf('Evaluating on training set...\n');
        [~, train_acc] = evaluate_svm(Xtr, Ytr, som, models, pca_model, cfg);
        fprintf('  Training accuracy: %.2f%%\n', train_acc * 100);

        % Evaluate on validation set
        fprintf('Evaluating on validation set...\n');
        [~, val_acc] = evaluate_svm(Xte, Yte, som, models, pca_model, cfg);
        fprintf('  Validation accuracy: %.2f%%\n', val_acc * 100);

        % Evaluate on real-world test
        fprintf('Evaluating on real-world test...\n');
        [realworld_pred_all, ~] = evaluate_svm(realworld_test.images, [], ...
                                                som, models, pca_model, cfg);
        % Extract in-vocabulary predictions
        realworld_pred = realworld_pred_all(realworld_test.invocab_positions);
        realworld_correct = sum(realworld_pred == realworld_test.labels);
        realworld_acc = realworld_correct / length(realworld_test.labels);
        fprintf('  Real-world accuracy: %.2f%% (%d/%d)\n', ...
                realworld_acc * 100, realworld_correct, length(realworld_test.labels));

        % Calculate gap
        train_val_gap = train_acc - val_acc;
        fprintf('  Train-to-val gap: %.2f%%\n\n', train_val_gap * 100);

        % Store results
        group_results.train_acc(cfg_idx) = train_acc;
        group_results.val_acc(cfg_idx) = val_acc;
        group_results.realworld_acc(cfg_idx) = realworld_acc;
        group_results.train_val_gap(cfg_idx) = train_val_gap;
        group_results.feature_dim(cfg_idx) = feature_dim;
        group_results.train_time(cfg_idx) = train_time;
    end

    all_results{group_idx} = group_results;

    % Save group results
    save(fullfile(output_dir, group.name, 'results.mat'), 'group_results', 'group');
end

%% Generate summary report
fprintf('\n');
fprintf('=========================================\n');
fprintf('Generating summary report...\n');
fprintf('=========================================\n\n');

generate_svm_hyper_summary_report(experiment_groups, all_results, output_dir);

fprintf('All experiments completed!\n');
fprintf('Results saved to: %s\n\n', output_dir);


%% Helper Functions

function [som, models, pca_model, feature_dim] = train_svm_pipeline(Xtr, Ytr, cfg)
    % Train complete SOM+BoW+SVM pipeline

    num_classes = 7;

    % Stage 1: Extract patches
    fprintf('  Stage 1: Extracting patches...\n');
    patches = extract_patches(Xtr, Ytr, cfg.patch_size, cfg.num_patch_samples, ...
        'normalize', true, 'verbose', false, 'min_patch_std', cfg.min_patch_std);

    % Stage 2: Train SOM
    fprintf('  Stage 2: Training SOM...\n');
    som = train_som_batch(patches, cfg.som_grid, cfg.som_iterations, ...
        'lr_init', cfg.som_lr_init, 'lr_final', cfg.som_lr_final, ...
        'sigma_init', cfg.som_sigma_init, 'sigma_final', cfg.som_sigma_final, ...
        'batch', cfg.som_batch, 'verbose', false);

    % Stage 3: Extract BoW features
    fprintf('  Stage 3: Extracting BoW features...\n');
    Ftr = extract_bow_features(Xtr, som, cfg.stride, ...
        'normalize', true, 'norm_type', 'l2', ...
        'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
        'spatial_pyramid', cfg.spatial_pyramid, 'pyramid_levels', cfg.pyramid_levels, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);

    fprintf('    Feature dimension (pre-PCA): %d\n', size(Ftr, 2));

    % Stage 4: Apply PCA
    if cfg.use_pca
        fprintf('  Stage 4: Applying PCA...\n');
        [Ftr_pca, ~, pca_model] = apply_pca(Ftr, [], cfg.pca_var);
        Ftr = Ftr_pca;
        fprintf('    Feature dimension (post-PCA): %d\n', size(Ftr, 2));
    else
        pca_model = [];
    end

    feature_dim = size(Ftr, 2);

    % Stage 5: Train SVM
    fprintf('  Stage 5: Training SVM...\n');
    models = trainMulticlassSVM(Ftr, Ytr, num_classes, cfg.C, ...
        'max_epochs', cfg.svm_epochs, 'lr', cfg.svm_lr, 'verbose', false);
end


function [predictions, accuracy] = evaluate_svm(X, Y, som, models, pca_model, cfg)
    % Evaluate SVM on given data

    % Handle cell array (from real-world test data)
    if iscell(X)
        % Convert cell array to 4D array
        num_images = length(X);
        X_array = zeros(64, 64, 1, num_images);
        for i = 1:num_images
            X_array(:, :, 1, i) = X{i};
        end
        % Apply polarity correction for real-world data
        X_array = 1 - X_array;
        X = X_array;
    end

    % Extract BoW features
    F = extract_bow_features(X, som, cfg.stride, ...
        'normalize', true, 'norm_type', 'l2', ...
        'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
        'spatial_pyramid', cfg.spatial_pyramid, 'pyramid_levels', cfg.pyramid_levels, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);

    % Apply PCA if needed
    if cfg.use_pca && ~isempty(pca_model)
        F = (F - pca_model.mu) * pca_model.coeff;
    end

    % Predict
    predictions = predictMulticlassSVM(models, F);

    % Calculate accuracy
    if isempty(Y)
        accuracy = NaN;
    else
        accuracy = mean(predictions == Y);
    end
end


function generate_svm_hyper_summary_report(experiment_groups, all_results, output_dir)
    % Generate summary report comparing all hyperparameters

    % Create summary table
    fid = fopen(fullfile(output_dir, 'summary.txt'), 'w');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'SVM Hyperparameter Sensitivity Summary\n');
    fprintf(fid, '========================================\n\n');

    for group_idx = 1:length(experiment_groups)
        group = experiment_groups{group_idx};
        results = all_results{group_idx};

        fprintf(fid, 'Parameter: %s\n', group.param_name);
        fprintf(fid, '-----------------------------------\n');
        fprintf(fid, 'Value\tTrain\tVal\tReal\tTrain-Val Gap\tFeat Dim\n');

        for i = 1:length(results.param_values)
            fprintf(fid, '%s\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%d\n', ...
                    group.value_labels{i}, ...
                    results.train_acc(i) * 100, ...
                    results.val_acc(i) * 100, ...
                    results.realworld_acc(i) * 100, ...
                    results.train_val_gap(i) * 100, ...
                    results.feature_dim(i));
        end
        fprintf(fid, '\n');
    end

    fclose(fid);
    fprintf('Summary report saved to: %s\n', fullfile(output_dir, 'summary.txt'));
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
