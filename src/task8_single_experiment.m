% task8_single_experiment.m
% Single experiment wrapper for Task 8 sensitivity analysis
%
% Input:
%   exp_config: cell array {exp_id, exp_name, aug_trans, aug_rot, aug_scale, lr, lr_method, batch_size}
%               OR struct with fields
%
% Output:
%   results: struct with experiment results

function results = task8_single_experiment(exp_config)
    % Parse experiment configuration
    if iscell(exp_config)
        config = struct();
        config.exp_id = exp_config{1};
        config.exp_name = exp_config{2};
        config.aug_trans = exp_config{3};
        config.aug_rot = exp_config{4};
        config.aug_scale = exp_config{5};
        config.lr = exp_config{6};
        config.lr_method = exp_config{7};
        config.batch_size = exp_config{8};
    else
        config = exp_config;
    end

    % Get project root
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    cd(project_root);

    % Add paths
    addpath(genpath('src/core'));
    addpath(genpath('src/utils'));

    % Create output directory
    % Special case: exp00 (baseline) saves to output/task7_1 for consistency with main task
    if strcmp(config.exp_id, 'exp00')
        output_base = fullfile('output', 'task7_1');
        % Use timestamped subdirectory
        date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
        output_dir = fullfile(output_base, char(date_prefix));
    else
        output_dir = fullfile('output', 'task8', config.exp_id);
    end

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    log_path = string(output_dir) + "/";

    % Setup logging
    diary_file = fullfile(output_dir, 'training_log.txt');
    diary(diary_file);
    diary on;

    fprintf('========================================\n');
    fprintf('Task 8 Experiment: %s (%s)\n', config.exp_id, config.exp_name);
    fprintf('========================================\n');
    fprintf('Configuration:\n');
    fprintf('  Augmentation: trans=%.3f, rot=[%d %d], scale=[%.2f %.2f]\n', ...
        config.aug_trans, config.aug_rot(1), config.aug_rot(2), ...
        config.aug_scale(1), config.aug_scale(2));
    fprintf('  Learning rate: %.3f (%s schedule)\n', config.lr, config.lr_method);
    fprintf('  Batch size: %d\n', config.batch_size);
    fprintf('  Output: %s\n', output_dir);
    fprintf('========================================\n\n');

    %% Dataset Configuration
    data_path = "data/";
    dataset_option.load_raw = true;
    dataset_option.shuffle = true;
    dataset_option.img_dim = 124;  % Fixed for all experiments
    dataset_option.train_ratio = 0.75;
    dataset_option.save = false;  % Don't resave, use existing

    % Configure data augmentation based on experiment
    has_augmentation = (config.aug_trans > 0) || ...
                      (config.aug_rot(1) ~= 0 || config.aug_rot(2) ~= 0) || ...
                      (config.aug_scale(1) ~= 1 || config.aug_scale(2) ~= 1);

    dataset_option.apply_rand_tf = has_augmentation;
    random_trans.prob = 0.5;
    random_trans.trans_ratio = config.aug_trans;
    random_trans.rot_range = config.aug_rot;
    random_trans.scale_ratio = config.aug_scale;
    dataset_option.rand_tf = random_trans;

    %% Load Dataset
    fprintf('Loading dataset...\n');
    % Load from pre-saved files for speed
    load('data/train.mat', 'data_train', 'labels_train');
    load('data/test.mat', 'data_test', 'labels_test');

    fprintf('Dataset loaded:\n');
    fprintf('  Training samples: %d\n', size(data_train, 4));
    fprintf('  Test samples: %d\n\n', size(data_test, 4));

    % Convert labels to 1-indexed
    labels_train = labels_train + 1;
    labels_test = labels_test + 1;

    %% Define CNN Architecture (Fixed baseline architecture)
    cnn.layers = {
        struct('type', 'input')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 64, 'poolDim', 2, 'actiFunc', 'relu')
        struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.3)
        struct('type', 'output', 'softmax', 1)
    };

    fprintf('CNN Architecture: 124x124 -> Conv(5x5,16) -> Conv(5x5,32) -> Conv(5x5,64) -> FC(128) -> Softmax(7)\n\n');

    %% Training Hyperparameters
    options.epochs = 30;
    options.minibatch = config.batch_size;
    options.lr_max = config.lr;
    options.lr = config.lr;
    options.lr_min = 1e-5;
    options.lr_method = config.lr_method;
    options.lr_duty = 20;  % For cyclic schedules
    options.momentum = 0.9;
    options.log_path = log_path;
    options.l2_penalty = 0.01;
    options.use_l2 = false;
    options.save_best_acc_model = true;
    options.train_mode = true;

    % Add step/exp specific parameters
    if strcmp(config.lr_method, 'step')
        options.lr_step_size = 10;
        options.lr_gamma = 0.1;
    elseif strcmp(config.lr_method, 'exp')
        options.lr_decay_rate = 0.95;
    end

    num_train_samples = size(data_train, 4);
    iter_per_epoch = max(1, floor((num_train_samples - options.minibatch) / options.minibatch) + 1);
    total_iter = iter_per_epoch * options.epochs;
    options.total_iter = total_iter;

    fprintf('Training Configuration:\n');
    fprintf('  Epochs: %d\n', options.epochs);
    fprintf('  Batch size: %d\n', options.minibatch);
    fprintf('  Learning rate: %.4f -> %.1e (%s)\n', options.lr_max, options.lr_min, options.lr_method);
    fprintf('  Momentum: %.2f\n', options.momentum);
    fprintf('  Dropout: 0.3\n');
    fprintf('  Total iterations: %d\n\n', total_iter);

    %% Initialize and Train CNN
    numClasses = 7;
    fprintf('Initializing network...\n');
    cnn = initModelParams(cnn, data_train, numClasses);

    fprintf('\n========== TRAINING START ==========\n');
    train_start_time = tic;
    cnn = learn(cnn, data_train, labels_train, data_test, labels_test, options);
    training_time = toc(train_start_time);
    fprintf('========== TRAINING COMPLETE ==========\n');
    fprintf('Training time: %.2f seconds (%.2f minutes)\n\n', training_time, training_time/60);

    %% Evaluate on Test Set
    fprintf('Final evaluation on test set...\n');
    [preds, ~] = predict(cnn, data_test);

    % Convert back to 0-indexed
    preds = preds - 1;
    labels_test_eval = labels_test - 1;

    test_acc = sum(preds == labels_test_eval) / length(preds);
    fprintf('Final test accuracy: %.4f (%.2f%%)\n\n', test_acc, test_acc * 100);

    % Per-class accuracy
    class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
    fprintf('Per-class accuracy:\n');
    per_class_acc = zeros(numClasses, 1);
    for i = 0:numClasses-1
        idx = (labels_test_eval == i);
        class_acc = sum(preds(idx) == labels_test_eval(idx)) / sum(idx);
        per_class_acc(i+1) = class_acc;
        fprintf('  Class %s: %.4f (%.2f%%) [%d samples]\n', ...
            class_names{i+1}, class_acc, class_acc*100, sum(idx));
    end

    % Train accuracy
    [preds_train, ~] = predict(cnn, data_train);
    preds_train = preds_train - 1;
    labels_train_eval = labels_train - 1;
    train_acc = sum(preds_train == labels_train_eval) / length(preds_train);
    fprintf('\nTrain accuracy: %.4f (%.2f%%)\n', train_acc, train_acc * 100);
    fprintf('Train-test gap: %.4f\n', train_acc - test_acc);

    %% Save Results
    fprintf('\n========================================\n');
    fprintf('Saving results to: %s\n', output_dir);

    % Save model
    save(log_path + "cnn_final.mat", 'cnn');
    save(log_path + "predictions.mat", 'preds', 'labels_test_eval');

    % Save configuration
    save(log_path + "config.mat", 'config');
    config_json = jsonencode(config);
    fid = fopen(log_path + "config.json", 'w');
    fprintf(fid, '%s', config_json);
    fclose(fid);

    % Save hyperparameters
    save(log_path + "options.mat", 'options');

    % Save results summary
    results = struct();
    results.exp_id = config.exp_id;
    results.exp_name = config.exp_name;
    results.test_acc = test_acc;
    results.train_acc = train_acc;
    results.train_test_gap = train_acc - test_acc;
    results.training_time = training_time;
    results.per_class_acc = per_class_acc;
    results.config = config;

    save(log_path + "results.mat", 'results');

    % Save text summary
    fileID = fopen(log_path + "results.txt", 'w');
    fprintf(fileID, 'Task 8 Experiment Results\n');
    fprintf(fileID, '========================================\n');
    fprintf(fileID, 'Experiment: %s (%s)\n\n', config.exp_id, config.exp_name);
    fprintf(fileID, 'Configuration:\n');
    fprintf(fileID, '  Augmentation: trans=%.3f, rot=[%d %d], scale=[%.2f %.2f]\n', ...
        config.aug_trans, config.aug_rot(1), config.aug_rot(2), ...
        config.aug_scale(1), config.aug_scale(2));
    fprintf(fileID, '  Learning rate: %.3f (%s)\n', config.lr, config.lr_method);
    fprintf(fileID, '  Batch size: %d\n\n', config.batch_size);
    fprintf(fileID, 'Results:\n');
    fprintf(fileID, '  Test accuracy: %.4f (%.2f%%)\n', test_acc, test_acc * 100);
    fprintf(fileID, '  Train accuracy: %.4f (%.2f%%)\n', train_acc, train_acc * 100);
    fprintf(fileID, '  Train-test gap: %.4f\n', train_acc - test_acc);
    fprintf(fileID, '  Training time: %.2f sec (%.2f min)\n\n', training_time, training_time/60);
    fprintf(fileID, 'Per-class accuracy:\n');
    for i = 0:numClasses-1
        fprintf(fileID, '  %s: %.4f (%.2f%%)\n', class_names{i+1}, per_class_acc(i+1), per_class_acc(i+1)*100);
    end
    fclose(fileID);

    fprintf('Results saved successfully!\n');
    fprintf('========================================\n\n');

    % Close diary
    diary off;

    % Display summary
    fprintf('Experiment %s completed:\n', config.exp_id);
    fprintf('  Test Acc: %.2f%%  |  Train Acc: %.2f%%  |  Gap: %.2f%%  |  Time: %.1f min\n', ...
        test_acc*100, train_acc*100, (train_acc-test_acc)*100, training_time/60);
end
