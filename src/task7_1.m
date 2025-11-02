% Task 7.1: CNN-based Character Classification
% Design a CNN to classify characters from the microchip label image
% Dataset: dataset_2025 with 7 classes (0, 4, 7, 8, A, D, H)
% Split: 75% training, 25% validation

clear all; %#ok<CLALL>
close all;

% Add paths
addpath('core/network');
addpath('core/image_process');
addpath('utils');

% Create timestamp for logging
date_prefix = string(datetime('now', 'Format', 'MM-dd_HH-mm-ss'));
log_path = "../output/task7_1/" + date_prefix + "/";
if ~exist(log_path, 'dir')
    mkdir(log_path);
end

fprintf('=== Task 7.1: CNN Character Classification ===\n');
fprintf('Log directory: %s\n\n', log_path);

%% Dataset Configuration
load_from_file = false;  % Set to true after first run to speed up
data_path = "../data/";

dataset_option.load_raw = true;
dataset_option.shuffle = true;
dataset_option.img_dim = 124;  % Resize to 124x124 for CNN
dataset_option.train_ratio = 0.75;
dataset_option.save = true;
dataset_option.apply_rand_tf = false;  % Disable for now - requires Image Processing Toolbox

% Random transformation for data augmentation
random_trans.prob = 0.5;          % 50% chance to apply transform
random_trans.trans_ratio = 0.01;  % Max 1% translation
random_trans.rot_range = [-25 25]; % Rotation range in degrees
random_trans.scale_ratio = [1 1];  % No scaling
dataset_option.rand_tf = random_trans;

%% Load Dataset
fprintf('Loading dataset...\n');
if ~load_from_file
    [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_option);
else
    fprintf('Loading from saved files...\n');
    load("../data/train.mat");
    load("../data/test.mat");
end

fprintf('Dataset loaded:\n');
fprintf('  Training samples: %d\n', size(data_train, 4));
fprintf('  Test samples: %d\n\n', size(data_test, 4));

% Convert labels from 0-indexed to 1-indexed for MATLAB
% Dataset uses 0-6, but MATLAB sparse() requires 1-7
labels_train = labels_train + 1;
labels_test = labels_test + 1;
fprintf('Labels converted to 1-indexed (1-7) for training\n\n');

%% Define CNN Architecture
% Improved Architecture: 124x124 -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3 -> Pool3 -> FC1 -> Output
% 124x124 -> (conv 5x5, 16 filters) -> 120x120x16 -> (pool 2x2) -> 60x60x16
%         -> (conv 5x5, 32 filters) -> 56x56x32 -> (pool 2x2) -> 28x28x32
%         -> (conv 5x5, 64 filters) -> 24x24x64 -> (pool 2x2) -> 12x12x64 = 9216
%         -> FC(128) -> Softmax(7)

cnn.layers = {
    struct('type', 'input')  % Input layer: 124x124x1
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 64, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.3)
    struct('type', 'output', 'softmax', 1)
};

fprintf('Improved CNN Architecture:\n');
fprintf('  Input: 124x124x1\n');
fprintf('  Conv1: 5x5x16, ReLU, MaxPool 2x2 -> 60x60x16\n');
fprintf('  Conv2: 5x5x32, ReLU, MaxPool 2x2 -> 28x28x32\n');
fprintf('  Conv3: 5x5x64, ReLU, MaxPool 2x2 -> 12x12x64\n');
fprintf('  FC1: 9216 -> 128, ReLU, Dropout(0.3)\n');
fprintf('  Output: 128 -> 7, Softmax\n\n');

%% Training Hyperparameters
options.epochs = 30;
options.minibatch = 128;
options.lr_max = 0.1;
options.lr = options.lr_max;
options.lr_min = 1e-5;
options.lr_method = 'linear';  % Linear decay
options.lr_duty = 20;          % Epochs per cycle (for cosine/cyclic)
options.momentum = 0.9;
options.log_path = log_path;
options.l2_penalty = 0.01;
options.use_l2 = false;        % L2 regularization
options.save_best_acc_model = true;
options.train_mode = true;

total_iter = round(floor(size(data_train, 4) / options.minibatch) * options.epochs);
options.total_iter = total_iter;

fprintf('Training Configuration:\n');
fprintf('  Epochs: %d\n', options.epochs);
fprintf('  Batch size: %d\n', options.minibatch);
fprintf('  Learning rate: %.4f (max) -> %.1e (min)\n', options.lr_max, options.lr_min);
fprintf('  LR schedule: %s\n', options.lr_method);
fprintf('  Momentum: %.2f\n', options.momentum);
fprintf('  Dropout: 0.3 (FC1)\n');
fprintf('  Total iterations: %d\n\n', total_iter);

%% Initialize and Train CNN
numClasses = max(labels_train) + 1 - min(labels_train);
fprintf('Number of classes: %d\n', numClasses);
fprintf('Initializing network parameters...\n');
cnn = initModelParams(cnn, data_train, numClasses);

fprintf('\nStarting training...\n');
fprintf('========================================\n');
tic;
cnn = learn(cnn, data_train, labels_train, data_test, labels_test, options);
training_time = toc;
fprintf('========================================\n');
fprintf('Training completed in %.2f seconds (%.2f minutes)\n\n', training_time, training_time/60);

%% Evaluate on Test Set
fprintf('Evaluating on test set...\n');
[preds, ~] = predict(cnn, data_test);

% Convert predictions and labels back to 0-indexed for evaluation
preds = preds - 1;
labels_test_eval = labels_test - 1;

test_acc = sum(preds == labels_test_eval) / length(preds);
fprintf('Final test accuracy: %.4f (%.2f%%)\n\n', test_acc, test_acc * 100);

% Per-class accuracy
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
fprintf('Per-class accuracy:\n');
for i = 0:numClasses-1
    idx = (labels_test_eval == i);
    class_acc = sum(preds(idx) == labels_test_eval(idx)) / sum(idx);
    fprintf('  Class %s: %.4f (%.2f%%) [%d samples]\n', ...
        class_names{i+1}, class_acc, class_acc*100, sum(idx));
end

%% Save Results
fprintf('\nSaving results to: %s\n', log_path);

% Save model
save(log_path + "cnn.mat", 'cnn');
save(log_path + "predictions.mat", 'preds', 'labels_test_eval');
save(log_path + "hyper_params.mat", 'options', 'dataset_option');

% Save text results
fileID = fopen(log_path + "results.txt", 'w');
fprintf(fileID, 'Task 7.1: CNN Character Classification\n');
fprintf(fileID, '========================================\n\n');
fprintf(fileID, 'Final test accuracy: %.4f (%.2f%%)\n\n', test_acc, test_acc * 100);
fprintf(fileID, 'Per-class accuracy:\n');
for i = 0:numClasses-1
    idx = (labels_test_eval == i);
    class_acc = sum(preds(idx) == labels_test_eval(idx)) / sum(idx);
    fprintf(fileID, '  Class %s: %.4f (%.2f%%) [%d samples]\n', ...
        class_names{i+1}, class_acc, class_acc*100, sum(idx));
end
fprintf(fileID, '\nTraining time: %.2f seconds (%.2f minutes)\n', training_time, training_time/60);
fclose(fileID);

% Save hyperparameters as JSON
json_opts = jsonencode(options);
fid = fopen(log_path + "hyper_params.json", 'w');
fprintf(fid, '%s', json_opts);
fclose(fid);

json_dataset = jsonencode(dataset_option);
fid = fopen(log_path + "dataset_option.json", 'w');
fprintf(fid, '%s', json_dataset);
fclose(fid);

% Save model architecture as JSON
model_json = model2json(cnn);
fid = fopen(log_path + "model.json", 'w');
fprintf(fid, '%s', model_json);
fclose(fid);

fprintf('\n=== Task 7.1 Complete ===\n');
