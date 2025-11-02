% Regenerate dataset with correct 0-indexed labels
clear; close all;

% Resolve directories
script_dir = fileparts(mfilename('fullpath'));
src_dir = fileparts(script_dir);
project_root = fileparts(src_dir);

% Preserve caller working directory
orig_dir = pwd;
cleanup = onCleanup(@() cd(orig_dir));

% Operate from src so legacy relative paths in utilities remain valid
cd(src_dir);
addpath(fullfile(src_dir, 'utils'));

data_path = "../data/";  % Relative to src directory (expected by loadDataset)

dataset_option.load_raw = true;
dataset_option.shuffle = true;
dataset_option.img_dim = 124;
dataset_option.train_ratio = 0.75;
dataset_option.save = true;
dataset_option.apply_rand_tf = false;

fprintf('=== Regenerating Dataset with Correct Labels ===\n');
[data_train, labels_train, data_test, labels_test] = loadDataset(data_path, dataset_option);

fprintf('\nDataset regenerated:\n');
fprintf('  Training samples: %d\n', size(data_train, 4));
fprintf('  Test samples: %d\n', size(data_test, 4));

fprintf('\nTrain set label distribution:\n');
for i = 0:6
    fprintf('  Class %d: %d samples\n', i, sum(labels_train == i));
end

fprintf('\nTest set label distribution:\n');
for i = 0:6
    fprintf('  Class %d: %d samples\n', i, sum(labels_test == i));
end

fprintf('\nLabel range:\n');
fprintf('  Train: [%d, %d]\n', min(labels_train), max(labels_train));
fprintf('  Test: [%d, %d]\n', min(labels_test), max(labels_test));

fprintf('\n✓ Dataset regeneration complete!\n');
