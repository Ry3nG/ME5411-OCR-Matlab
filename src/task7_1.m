% Task 7.1 Baseline Runner

clear all; %#ok<CLALL>
close all;

%% Setup
fprintf('\n');
fprintf('========================================\n');
fprintf('   Task 7.1: Baseline Experiment Runner   \n');
fprintf('========================================\n');
fprintf('Started at: %s\n', datetime('now'));
fprintf('========================================\n\n');

% Determine project root from this script location
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Ensure shared utilities are on the MATLAB path
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

%% Baseline configuration (matches Task 8 exp00)
config = struct();
config.exp_id = 'exp00';
config.exp_name = 'baseline';
config.aug_trans = 0;
config.aug_rot = [0 0];
config.aug_scale = [1 1];
config.lr = 0.1;
config.lr_method = 'linear';
config.batch_size = 128;
config.force_regenerate_dataset = false;  % Set true to rebuild cached splits

fprintf('Prepared baseline configuration (exp00):\n');
fprintf('  Augmentation: translation=%.2f, rotation=[%.0f %.0f], scale=[%.2f %.2f]\n', ...
    config.aug_trans, config.aug_rot(1), config.aug_rot(2), ...
    config.aug_scale(1), config.aug_scale(2));
fprintf('  Learning rate: %.3f (%s schedule)\n', config.lr, config.lr_method);
fprintf('  Batch size: %d\n', config.batch_size);

%% Prepare output directory and logging
output_root = fullfile('output', 'task7_1');
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
output_dir = fullfile(output_root, char(date_prefix));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
log_path = string(output_dir) + "/";

diary_file = fullfile(output_dir, 'training_log.txt');
diary(diary_file);
diary on;

fprintf('Output directory: %s\n', output_dir);
fprintf('Training log: %s\n\n', diary_file);

cleanup_diary = onCleanup(@() diary('off'));

try
    %% Dataset configuration (saved for reproducibility)
    dataset_option = struct();
    dataset_option.load_raw = true;
    dataset_option.shuffle = true;
    dataset_option.img_dim = 64;
    dataset_option.train_ratio = 0.75;
    dataset_option.apply_rand_tf = false;
    random_trans = struct('prob', 0.5, ...
                          'trans_ratio', config.aug_trans, ...
                          'rot_range', config.aug_rot, ...
                          'scale_ratio', config.aug_scale);
    dataset_option.rand_tf = random_trans;

    data_dir = "data";
    train_cache = fullfile(char(data_dir), 'train.mat');
    test_cache = fullfile(char(data_dir), 'test.mat');

    regenerate_needed = config.force_regenerate_dataset || ...
        ~exist(train_cache, 'file') || ~exist(test_cache, 'file');

    if regenerate_needed
        fprintf('Cached dataset missing or regeneration requested.\n');
        fprintf('Generating dataset splits via loadDataset...\n');
        dataset_option.save = true;
        [data_train, labels_train, data_test, labels_test] = loadDataset(data_dir, dataset_option);
    else
        dataset_option.save = false;
        fprintf('Loading dataset from cache...\n');
        load(train_cache, 'data_train', 'labels_train');
        load(test_cache, 'data_test', 'labels_test');
    end

    fprintf('Dataset ready:\n');
    fprintf('  Training samples: %d\n', size(data_train, 4));
    fprintf('  Test samples: %d\n\n', size(data_test, 4));

    % Convert labels to 1-indexed for MATLAB operations
    labels_train = labels_train + 1;
    labels_test = labels_test + 1;

    %% Define CNN architecture (efficiency-focused)
    cnn.layers = {
        struct('type', 'input')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4,  'poolDim', 4, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 4, 'numFilters', 8,  'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 3, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
        struct('type', 'Linear', 'hiddenUnits', 50,  'actiFunc', 'relu')
        struct('type', 'output', 'softmax', 1)
    };

    fprintf(['CNN Architecture (efficiency-first):\n' ...
        '  64x64 -> Conv(5x5,x4) + Pool(4) -> 15x15x4\n' ...
        '        -> Conv(4x4,x8) + Pool(2) -> 6x6x8\n' ...
        '        -> Conv(3x3,x16) + Pool(2) -> 2x2x16\n' ...
        '        -> FC(100, dropout 0.2) -> FC(50) -> Softmax(7)\n\n']);

    %% Training hyperparameters
    options = struct();
    options.epochs = 30;
    options.minibatch = config.batch_size;
    options.lr_max = config.lr;
    options.lr = config.lr;
    options.lr_min = 1e-5;
    options.lr_method = config.lr_method;
    options.lr_duty = 20;
    options.momentum = 0.9;
    options.log_path = log_path;
    options.l2_penalty = 0.01;
    options.use_l2 = false;
    options.save_best_acc_model = true;
    options.train_mode = true;

    if strcmp(options.lr_method, 'step')
        options.lr_step_size = 10;
        options.lr_gamma = 0.1;
    elseif strcmp(options.lr_method, 'exp')
        options.lr_decay_rate = 0.95;
    end

    num_train_samples = size(data_train, 4);
    iter_per_epoch = max(1, floor((num_train_samples - options.minibatch) / options.minibatch) + 1);
    options.total_iter = iter_per_epoch * options.epochs;

    fprintf('Training configuration:\n');
    fprintf('  Epochs: %d\n', options.epochs);
    fprintf('  Batch size: %d\n', options.minibatch);
    fprintf('  Learning rate: %.4f -> %.1e (%s)\n', options.lr_max, options.lr_min, options.lr_method);
    fprintf('  Momentum: %.2f\n', options.momentum);
    fprintf('  Total iterations: %d\n\n', options.total_iter);

    %% Initialize and train CNN
    numClasses = 7;
    fprintf('Initializing network...\n');
    cnn = initModelParams(cnn, data_train, numClasses);

    fprintf('\n========== TRAINING START ==========\n');
    train_start_time = tic;
    cnn = learn(cnn, data_train, labels_train, data_test, labels_test, options);
    training_time = toc(train_start_time);
    fprintf('========== TRAINING COMPLETE ==========\n');
    fprintf('Training time: %.2f seconds (%.2f minutes)\n\n', training_time, training_time/60);

    %% Evaluate on test set
    fprintf('Final evaluation on test set...\n');
    [preds_test, ~] = predict(cnn, data_test);
    preds_test = preds_test - 1;
    labels_test_eval = labels_test - 1;

    test_acc = sum(preds_test == labels_test_eval) / length(preds_test);
    fprintf('Test accuracy: %.4f (%.2f%%)\n', test_acc, test_acc * 100);

    class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
    per_class_acc = zeros(numClasses, 1);
    fprintf('Per-class accuracy:\n');
    for i = 0:numClasses-1
        idx = (labels_test_eval == i);
        class_acc = sum(preds_test(idx) == labels_test_eval(idx)) / sum(idx);
        per_class_acc(i+1) = class_acc;
        fprintf('  Class %s: %.4f (%.2f%%) [%d samples]\n', ...
            class_names{i+1}, class_acc, class_acc*100, sum(idx));
    end

    [preds_train, ~] = predict(cnn, data_train);
    preds_train = preds_train - 1;
    labels_train_eval = labels_train - 1;
    train_acc = sum(preds_train == labels_train_eval) / length(preds_train);
    train_test_gap = train_acc - test_acc;

    fprintf('\nTrain accuracy: %.4f (%.2f%%)\n', train_acc, train_acc * 100);
    fprintf('Train-test gap: %.4f (%.2f%%)\n\n', train_test_gap, train_test_gap * 100);

    %% Save artifacts
    fprintf('========================================\n');
    fprintf('Saving artifacts to: %s\n', output_dir);

    save(log_path + "cnn_final.mat", 'cnn');
    save(log_path + "predictions.mat", 'preds_test', 'labels_test_eval');
    save(log_path + "options.mat", 'options');
    save(log_path + "dataset_option.mat", 'dataset_option');

    config.output_dir = output_dir;
    save(log_path + "config.mat", 'config');
    fid = fopen(log_path + "config.json", 'w');
    fprintf(fid, '%s', jsonencode(config));
    fclose(fid);

    fid = fopen(log_path + "options.json", 'w');
    fprintf(fid, '%s', jsonencode(options));
    fclose(fid);

    fid = fopen(log_path + "dataset_option.json", 'w');
    fprintf(fid, '%s', jsonencode(dataset_option));
    fclose(fid);

    results = struct();
    results.exp_id = config.exp_id;
    results.exp_name = config.exp_name;
    results.test_acc = test_acc;
    results.train_acc = train_acc;
    results.train_test_gap = train_test_gap;
    results.training_time = training_time;
    results.per_class_acc = per_class_acc;
    results.config = config;
    save(log_path + "results.mat", 'results');

    fileID = fopen(log_path + "results.txt", 'w');
    fprintf(fileID, 'Task 7.1 Baseline Results\n');
    fprintf(fileID, '========================================\n');
    fprintf(fileID, 'Generated: %s\n\n', datetime('now'));
    fprintf(fileID, 'Test accuracy: %.4f (%.2f%%)\n', test_acc, test_acc*100);
    fprintf(fileID, 'Train accuracy: %.4f (%.2f%%)\n', train_acc, train_acc*100);
    fprintf(fileID, 'Train-test gap: %.4f\n', train_test_gap);
    fprintf(fileID, 'Training time: %.2f sec (%.2f min)\n\n', training_time, training_time/60);
    fprintf(fileID, 'Per-class accuracy:\n');
    for i = 1:numClasses
        fprintf(fileID, '  %s: %.4f (%.2f%%)\n', class_names{i}, per_class_acc(i), per_class_acc(i)*100);
    end
    fclose(fileID);

    fprintf('Artifacts saved successfully.\n');
    fprintf('========================================\n\n');

    %% Final console summary
    fprintf('Baseline experiment complete.\n');
    fprintf('  Test Accuracy : %.2f%%\n', test_acc * 100);
    fprintf('  Train Accuracy: %.2f%%\n', train_acc * 100);
    fprintf('  Train-Test Gap: %.2f%%\n', train_test_gap * 100);
    fprintf('  Training Time : %.1f minutes\n', training_time / 60);
    fprintf('Artifacts directory: %s\n', output_dir);
    fprintf('  - cnn_final.mat, predictions.mat, results.mat\n');
    fprintf('  - config.json, options.json, dataset_option.json\n');
    fprintf('  - results.txt, training_log.txt\n');
    fprintf('\n✓ Task 7.1 baseline run finished.\n');

catch ME
    fprintf('\n✗ Baseline experiment failed with error:\n');
    fprintf('  %s\n', ME.message);
    rethrow(ME);
end

