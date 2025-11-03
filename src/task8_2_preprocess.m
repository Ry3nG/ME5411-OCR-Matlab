% TASK8_2 Targeted experiments for Task 8 pre-processing sensitivity
%
% Experiments cover geometry pre-processing variants requested by the spec:
%   exp01 - translation  ±2% shift
%   exp02 - rotation     ±15 degrees
%   exp03 - scale        0.9–1.1×
%   exp04 - all_aug      combination of the above
%
% Each variant reuses the Task 7.2 SOM→BoW→SVM pipeline with filtered patches.
% Results (accuracy, timing, feature dimension, etc.) are logged to
% output/task8_2/summary.csv and per-experiment subfolders.
%
% Usage:
%   matlab -batch "run('src/task8_2.m')"

function task8_2()
    rng(0, 'twister');

    % Resolve project root and add source paths
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    cd(project_root);
    addpath(genpath('src'));

    % Output root
    out_root = 'output/task8_2';
    if ~exist(out_root, 'dir')
        mkdir(out_root);
    end

    % Load Canvas dataset
    fprintf('=== Task 8.2: Geometry Pre-processing Study ===\n\n');
    load('data/train.mat', 'data_train', 'labels_train');
    load('data/test.mat', 'data_test', 'labels_test');
    Xtr_raw = data_train;
    Xte_raw = data_test;
    Ytr = labels_train + 1;
    Yte = labels_test + 1;

    % Pipeline configuration (aligned with Task 7.2)
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
    cfg.min_patch_std = 0.001;

    % Experiment definitions
    experiments = {
        struct('name', 'exp01_translation', 'label', 'translation', ...
               'trans_ratio', 0.02, 'rot_range', [], 'scale_range', [])
        struct('name', 'exp02_rotation', 'label', 'rotation', ...
               'trans_ratio', [], 'rot_range', [-15, 15], 'scale_range', [])
        struct('name', 'exp03_scale', 'label', 'scale', ...
               'trans_ratio', [], 'rot_range', [], 'scale_range', [0.9, 1.1])
        struct('name', 'exp04_all_aug', 'label', 'all_aug', ...
               'trans_ratio', 0.02, 'rot_range', [-15, 15], 'scale_range', [0.9, 1.1])
    };

    records = [];

    for idx = 1:numel(experiments)
        exp = experiments{idx};
        fprintf('--- %s (%s) ---\n', exp.name, exp.label);

        exp_dir = fullfile(out_root, exp.name);
        if ~exist(exp_dir, 'dir')
            mkdir(exp_dir);
        end

        % Apply geometry pre-processing
        fprintf('  Applying geometry transform to train/test stacks...\n');
        Xtr = apply_geometry_stack(Xtr_raw, exp, 100 + idx);
        Xte = apply_geometry_stack(Xte_raw, exp, 200 + idx);

        % Run pipeline
        metrics = run_pipeline(Xtr, Ytr, Xte, Yte, cfg, exp_dir, exp);
        metrics.experiment = exp.name;
        metrics.label = exp.label;
        metrics.trans_ratio = default_zero(exp.trans_ratio);
        metrics.rot_min = default_zero(exp.rot_range, 1);
        metrics.rot_max = default_zero(exp.rot_range, 2);
        metrics.scale_min = default_zero(exp.scale_range, 1);
        metrics.scale_max = default_zero(exp.scale_range, 2);

        records = [records; metrics]; %#ok<AGROW>
        fprintf('  Accuracy: %.2f%% | Train time: %.2fs | Test time: %.2fs | feat_dim=%d\n\n', ...
            metrics.accuracy * 100, metrics.train_time, metrics.test_time, metrics.feature_dim);
    end

    % Write summary CSV
    summary_path = fullfile(out_root, 'summary.csv');
    T = struct2table(records);
    writetable(T, summary_path);
    fprintf('Summary saved to %s\n', summary_path);
    fprintf('=== Task 8.2 experiments complete ===\n');
end

%% ------------------------------------------------------------------------
function data_out = apply_geometry_stack(data_in, exp, base_seed)
    data_out = zeros(size(data_in), 'uint8');
    [H, W, ~, N] = size(data_in);
    rng(base_seed, 'twister');

    for i = 1:N
        img = data_in(:, :, 1, i);
        img = apply_geometry(img, H, W, exp);
        data_out(:, :, 1, i) = img;
    end
end

function img_out = apply_geometry(img_in, H, W, exp)
    img = img_in;

    % Translation
    if ~isempty(exp.trans_ratio)
        max_shift = round(H * exp.trans_ratio);
        dx = randi([-max_shift, max_shift]);
        dy = randi([-max_shift, max_shift]);
        img = translate_image(img, dx, dy);
    end

    % Rotation
    if ~isempty(exp.rot_range)
        angle = exp.rot_range(1) + (exp.rot_range(2) - exp.rot_range(1)) * rand();
        img = rotate_image(img, angle, H, W);
    end

    % Scaling
    if ~isempty(exp.scale_range)
        scale = exp.scale_range(1) + (exp.scale_range(2) - exp.scale_range(1)) * rand();
        img = scale_image(img, scale, H, W);
    end

    img_out = uint8(img);
end

function img_out = translate_image(img, dx, dy)
    [H, W] = size(img);
    img_out = uint8(ones(H, W) * 255);

    src_x_start = max(1, 1 - dx);
    src_x_end = min(W, W - dx);
    src_y_start = max(1, 1 - dy);
    src_y_end = min(H, H - dy);

    dst_x_start = max(1, 1 + dx);
    dst_x_end = min(W, W + dx);
    dst_y_start = max(1, 1 + dy);
    dst_y_end = min(H, H + dy);

    if src_x_start <= src_x_end && src_y_start <= src_y_end
        img_out(dst_y_start:dst_y_end, dst_x_start:dst_x_end) = ...
            img(src_y_start:src_y_end, src_x_start:src_x_end);
    end
end

function img_out = rotate_image(img, angle_deg, H, W)
    img_d = double(img);
    theta = angle_deg * pi / 180;
    cosA = cos(theta);
    sinA = sin(theta);
    [X, Y] = meshgrid(1:W, 1:H);
    cx = (W + 1) / 2;
    cy = (H + 1) / 2;

    Xc = X - cx;
    Yc = Y - cy;

    Xs = cosA * Xc + sinA * Yc + cx;
    Ys = -sinA * Xc + cosA * Yc + cy;

    rotated = interp2(img_d, Xs, Ys, 'linear', 255);
    rotated(isnan(rotated)) = 255;
    img_out = uint8(rotated);
end

function img_out = scale_image(img, scale, H, W)
    scaled = imresize(img, scale, 'bilinear');
    scaled = uint8(scaled);
    [Hs, Ws] = size(scaled);

    if Hs >= H && Ws >= W
        % Center crop
        y_start = floor((Hs - H) / 2) + 1;
        x_start = floor((Ws - W) / 2) + 1;
        img_out = scaled(y_start:y_start+H-1, x_start:x_start+W-1);
    else
        % Center pad with white background
        img_out = uint8(ones(H, W) * 255);
        y_start = floor((H - Hs) / 2) + 1;
        x_start = floor((W - Ws) / 2) + 1;
        img_out(y_start:y_start+Hs-1, x_start:x_start+Ws-1) = scaled;
    end
end

function value = default_zero(arr, idx)
    if nargin < 2
        if isempty(arr)
            value = 0;
        else
            value = arr;
        end
    else
        if isempty(arr)
            value = 0;
        else
            value = arr(idx);
        end
    end
end

function metrics = run_pipeline(Xtr, Ytr, Xte, Yte, cfg, out_dir, exp)
    [H, W, ~, N_train] = size(Xtr);
    N_test = size(Xte, 4);
    num_classes = numel(unique(Ytr));
    fprintf('  Training samples: %d | Test samples: %d\n', N_train, N_test);

    % Patch sampling
    tic;
    patches = extract_patches(Xtr, Ytr, cfg.patch_size, cfg.num_patch_samples, ...
        'normalize', true, 'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_patches = toc;

    % SOM training
    tic;
    som = train_som_batch(patches, cfg.som_grid, cfg.som_iterations, ...
        'lr_init', cfg.som_lr_init, 'lr_final', cfg.som_lr_final, ...
        'sigma_init', cfg.som_sigma_init, 'sigma_final', cfg.som_sigma_final, ...
        'batch', cfg.som_batch, 'verbose', false);
    t_som = toc;

    save(fullfile(out_dir, 'som_model.mat'), 'som', 'exp');

    % Feature extraction
    tic;
    Ftr = extract_bow_features(Xtr, som, cfg.stride, ...
        'normalize', true, 'soft_voting', cfg.soft_voting, ...
        'sigma_bow', cfg.sigma_bow, 'spatial_pyramid', cfg.spatial_pyramid, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_train_feat = toc;

    tic;
    Fte = extract_bow_features(Xte, som, cfg.stride, ...
        'normalize', true, 'soft_voting', cfg.soft_voting, ...
        'sigma_bow', cfg.sigma_bow, 'spatial_pyramid', cfg.spatial_pyramid, ...
        'verbose', false, 'min_patch_std', cfg.min_patch_std);
    t_test_feat = toc;

    feat_dim_pre = size(Ftr, 2);

    % PCA
    if cfg.use_pca
        [Ftr, Fte, pca_model] = apply_pca(Ftr, Fte, cfg.pca_var);
        save(fullfile(out_dir, 'pca_model.mat'), 'pca_model');
    end
    feat_dim_post = size(Ftr, 2);

    % SVM training
    tic;
    models = trainMulticlassSVM(Ftr, Ytr, num_classes, cfg.C, ...
        'max_epochs', cfg.svm_epochs, 'lr', cfg.svm_lr, 'verbose', false);
    t_svm = toc;
    save(fullfile(out_dir, 'svm_models.mat'), 'models');

    % Prediction
    tic;
    predictions = predictMulticlassSVM(models, Fte);
    t_pred = toc;

    accuracy = mean(predictions == Yte);

    metrics = struct();
    metrics.accuracy = accuracy;
    metrics.train_time = t_patches + t_som + t_train_feat + t_svm;
    metrics.test_time = t_test_feat + t_pred;
    metrics.feature_dim = feat_dim_post;
    metrics.feature_dim_pre = feat_dim_pre;
    metrics.patches_collected = size(patches, 1);
    metrics.patch_time = t_patches;
    metrics.som_time = t_som;
    metrics.feat_train_time = t_train_feat;
    metrics.feat_test_time = t_test_feat;
    metrics.svm_time = t_svm;
    metrics.pred_time = t_pred;
    metrics.config = cfg;

    save(fullfile(out_dir, 'results.mat'), 'metrics', 'predictions', 'Yte', 'exp');
end
