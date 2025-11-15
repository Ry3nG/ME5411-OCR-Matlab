













% Quick verification script
clear; close all;

% Load baseline
baseline = load('data/train.mat');
fprintf('Baseline (train.mat):\n');
fprintf('  Format: ');
unique_vals = length(unique(baseline.data_train(:, :, 1, 1)));
if unique_vals == 2
    fprintf('BINARY (2 unique values)\n');
else
    fprintf('GRAYSCALE (%d unique values)\n', unique_vals);
end
fprintf('  Size: [%s]\n', mat2str(size(baseline.data_train)));

% Load noise (corrected)
noise = load('data/train_noise.mat');
fprintf('\nNoise (train_noise.mat - CORRECTED):\n');
fprintf('  Format: ');
unique_vals_noise = length(unique(noise.data_train(:, :, 1, 1)));
if unique_vals_noise == 2
    fprintf('BINARY (2 unique values) ❌ FAILED!\n');
else
    fprintf('GRAYSCALE (%d unique values) ✅ SUCCESS!\n', unique_vals_noise);
end
fprintf('  Size: [%s]\n', mat2str(size(noise.data_train)));

% Check match
if (unique_vals == 2 && unique_vals_noise > 2) || ...
   (unique_vals > 2 && unique_vals_noise == 2)
    fprintf('\n⚠️  Format MISMATCH detected!\n');
else
    fprintf('\n✅ Formats MATCH!\n');
end
