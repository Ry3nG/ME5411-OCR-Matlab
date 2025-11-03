% TASK8_2_HYPERPARAM Targeted hyperparameter sensitivity experiments
%
% This script fixes the best pre-processing configuration and probes the
% SOM→BoW→Linear-SVM hyperparameters individually:
%   1) Codebook size K            : [100, 144, 196]
%   2) Patch/stride combinations  : {[8,4], [6,2], [10,4]}
%   3) Voting scheme              : {'soft', 'hard'}
%   4) Spatial pyramid usage      : [true, false]
%   5) SVM regularization C       : [0.3, 1.0, 3.0]
%
% Each factor is varied while the others are kept at the Task 7.2 baseline.
% Results for every run are logged to output/task8_2/hyperparam.csv.
%
% Usage:
%   matlab -batch "run('src/task8_2_hyperparam.m')"

function task8_2_hyperparam()
    rng(0, 'twister');

    % Paths
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    cd(project_root);
    addpath(genpath('src'));

    out_root = 'output/task8_2';
    if ~exist(out_root, 'dir')
        mkdir(out_root);
    end

    % Fixed dataset (raw images without extra preprocessing)
    load('data/train.mat', 'data_train', 'labels_train');
    load('data/test.mat', 'data_test', 'labels_test');
    Xtr = data_train;
    Xte = data_test;
    Ytr = labels_train + 1;
    Yte = labels_test + 1;

    % Baseline configuration (in sync with Task 7.2)
    base_cfg = struct();
    base_cfg.patch_size = 8;
    base_cfg.num_patch_samples = 100000;
    base_cfg.som_grid = [10, 10];
    base_cfg.som_iterations = 50000;
    base_cfg.som_lr_init = 0.5;
    base_cfg.som_lr_final = 0.01;
    base_cfg.som_sigma_init = max(base_cfg.som_grid) / 2;
    base_cfg.som_sigma_final = 0.5;
    base_cfg.som_batch = 32;
    base_cfg.stride = 4;
    base_cfg.soft_voting = true;
    base_cfg.sigma_bow = 0.75;
    base_cfg.spatial_pyramid = true;
    base_cfg.C = 1.0;
    base_cfg.svm_epochs = 200;
    base_cfg.svm_lr = 0.01;
    base_cfg.use_pca = true;
    base_cfg.pca_var = 0.95;
    base_cfg.min_patch_std = 0.001;

    % Experiment grids
    grid.K = [100, 144, 196];
    grid.patch_stride = [8 4; 6 2; 10 4];
    grid.vote = {'soft', 'hard'};
    grid.spm = [true, false];
    grid.C = [0.3, 1.0, 3.0];

    % Prepare log file
    logfile = fullfile(out_root, 'hyperparam.csv');
    fid = fopen(logfile, 'w');
    header = ['factor,value,K,patch,stride,vote,spm,C,acc,' ...
              'train_sec,test_sec,feat_dim,feat_dim_pre,min_patch_std'];
    fprintf(fid, '%s\n', header);

    % 1) Codebook size sweep
    for val = grid.K
        cfg = base_cfg;
        cfg.som_grid = [round(sqrt(val)), round(sqrt(val))];
        cfg.som_sigma_init = max(cfg.som_grid) / 2;
        run_and_log('codebook', val, cfg, Xtr, Ytr, Xte, Yte, fid);
    end

    % 2) Patch/stride sweep
    for row = 1:size(grid.patch_stride, 1)
        cfg = base_cfg;
        cfg.patch_size = grid.patch_stride(row, 1);
        cfg.stride = grid.patch_stride(row, 2);
        run_and_log('patch_stride', sprintf('%d-%d', cfg.patch_size, cfg.stride), cfg, Xtr, Ytr, Xte, Yte, fid);
    end

    % 3) Voting scheme sweep
    for i = 1:numel(grid.vote)
        cfg = base_cfg;
        cfg.soft_voting = strcmp(grid.vote{i}, 'soft');
        run_and_log('voting', grid.vote{i}, cfg, Xtr, Ytr, Xte, Yte, fid);
    end

    % 4) Spatial pyramid sweep
    for val = grid.spm
        cfg = base_cfg;
        cfg.spatial_pyramid = val;
        run_and_log('spatial_pyramid', logical_to_str(val), cfg, Xtr, Ytr, Xte, Yte, fid);
    end

    % 5) SVM C sweep
    for val = grid.C
        cfg = base_cfg;
        cfg.C = val;
        run_and_log('svm_C', val, cfg, Xtr, Ytr, Xte, Yte, fid);
    end

    fclose(fid);
    fprintf('Hyperparameter results saved to %s\n', logfile);
end

%% ------------------------------------------------------------------------
function run_and_log(factor, value, cfg, Xtr, Ytr, Xte, Yte, fid)
    fprintf('\n[%s] value=%s\n', factor, to_string(value));

    % Extract patches
    tic;
    patches = extract_patches(Xtr, Ytr, cfg.patch_size, cfg.num_patch_samples, ...
        'normalize', true, 'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_patch = toc;

    % Train SOM
    tic;
    som = train_som_batch(patches, cfg.som_grid, cfg.som_iterations, ...
        'lr_init', cfg.som_lr_init, 'lr_final', cfg.som_lr_final, ...
        'sigma_init', cfg.som_sigma_init, 'sigma_final', cfg.som_sigma_final, ...
        'batch', cfg.som_batch, 'verbose', false);
    t_som = toc;

    % Feature extraction
    tic;
    Ftr = extract_bow_features(Xtr, som, cfg.stride, ...
        'normalize', true, 'soft_voting', cfg.soft_voting, ...
        'sigma_bow', cfg.sigma_bow, 'spatial_pyramid', cfg.spatial_pyramid, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_feat_tr = toc;

    tic;
    Fte = extract_bow_features(Xte, som, cfg.stride, ...
        'normalize', true, 'soft_voting', cfg.soft_voting, ...
        'sigma_bow', cfg.sigma_bow, 'spatial_pyramid', cfg.spatial_pyramid, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_feat_te = toc;

    feat_dim_pre = size(Ftr, 2);

    % PCA (optionally)
    if cfg.use_pca
        [Ftr, Fte, ~] = apply_pca(Ftr, Fte, cfg.pca_var);
    end
    feat_dim_post = size(Ftr, 2);

    % SVM
    tic;
    models = trainMulticlassSVM(Ftr, Ytr, numel(unique(Ytr)), cfg.C, ...
        'max_epochs', cfg.svm_epochs, 'lr', cfg.svm_lr, 'verbose', false);
    t_svm = toc;

    % Prediction
    tic;
    predictions = predictMulticlassSVM(models, Fte);
    t_pred = toc;

    accuracy = mean(predictions == Yte);
    train_time = t_patch + t_som + t_feat_tr + t_svm;
    test_time = t_feat_te + t_pred;

    fprintf('  Accuracy: %.2f%% | Train: %.2fs | Test: %.2fs | feat_dim=%d (pre=%d)\n', ...
        accuracy * 100, train_time, test_time, feat_dim_post, feat_dim_pre);

    fprintf(fid, '%s,%s,%d,%d,%d,%s,%d,%.2f,%.6f,%.3f,%.3f,%d,%d,%.4f\n', ...
        factor, to_string(value), som.num_neurons, cfg.patch_size, cfg.stride, ...
        bool_to_label(cfg.soft_voting), cfg.spatial_pyramid, cfg.C, accuracy, ...
        train_time, test_time, feat_dim_post, feat_dim_pre, cfg.min_patch_std);
end

%% ------------------------------------------------------------------------
function s = to_string(val)
    if ischar(val) || isstring(val)
        s = char(val);
    elseif islogical(val)
        s = logical_to_str(val);
    elseif isnumeric(val)
        if numel(val) == 1
            s = num2str(val);
        else
            s = mat2str(val);
        end
    else
        s = '<unknown>';
    end
end

function s = logical_to_str(val)
    if val
        s = 'true';
    else
        s = 'false';
    end
end

function lbl = bool_to_label(val)
    if val
        lbl = 'soft';
    else
        lbl = 'hard';
    end
end
