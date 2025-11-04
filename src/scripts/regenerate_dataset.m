% Regenerate dataset with custom functions (myImresize)
% This ensures train.mat and test.mat use only custom implementations

clear all; %#ok<CLALL>
close all;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts
src_dir = fileparts(script_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

fprintf('=== Regenerating Dataset with Custom Functions ===\n\n');

%% Dataset Configuration
data_path = "data/";

dataset_option.load_raw = true;  % Load from raw images
dataset_option.shuffle = true;
dataset_option.img_dim = 124;  % Resize to 124x124 for CNN
dataset_option.train_ratio = 0.75;
dataset_option.save = true;  % Save to train.mat and test.mat
dataset_option.apply_rand_tf = false;  % No data augmentation

% --- Ensure data save folder exists (relative path) ---
data_save_dir = fullfile(project_root, 'data');
if ~exist(data_save_dir, 'dir')
    fprintf('[INFO] Creating data save directory: %s\n', data_save_dir);
    mkdir(data_save_dir);
end

%% Load Dataset (this will use custom myImresize)
fprintf('Loading and processing dataset with custom functions...\n');
fprintf('[INFO] Dataset .mat files will be saved to: %s\n', data_save_dir);

tic;
[data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_option);
elapsed = toc;

fprintf('\nDataset regenerated successfully!\n');
fprintf('  Training samples: %d\n', size(data_train, 4));
fprintf('  Test samples: %d\n', size(data_test, 4));
fprintf('  Processing time: %.2f seconds\n', elapsed);

% Verify data
fprintf('\nData verification:\n');
fprintf('  train data size: %s\n', mat2str(size(data_train)));
fprintf('  train labels size: %s\n', mat2str(size(labels_train)));
fprintf('  test data size: %s\n', mat2str(size(data_test)));
fprintf('  test labels size: %s\n', mat2str(size(labels_test)));
fprintf('  train data range: [%.4f, %.4f]\n', min(data_train(:)), max(data_train(:)));
fprintf('  test data range: [%.4f, %.4f]\n', min(data_test(:)), max(data_test(:)));

fprintf('\n✓ Dataset files (train.mat, test.mat) are now using custom implementations!\n');
fprintf('✓ Ready for task7_1 training with load_from_file = true\n');
