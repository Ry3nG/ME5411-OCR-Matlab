% visualize_dataset.m
% Generate visualizations for the dataset preparation section
% Outputs:
% 1. Sample images from each class (training set)
% 2. Class distribution bar chart
% 3. Grid of random samples from training and validation sets

clear; clc;

% Add paths
addpath('utils');

% Create output directory
outputDir = '../output/dataset_visualization';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('Loading datasets...\n');
% Load datasets
load('../data/train.mat', 'data_train', 'labels_train');
load('../data/test.mat', 'data_test', 'labels_test');

% Reshape to 3D arrays (H x W x N)
trainImages = squeeze(data_train);
trainLabels = labels_train;
testImages = squeeze(data_test);
testLabels = labels_test;

% Character classes
classes = {'0', '4', '7', '8', 'A', 'D', 'H'};
numClasses = length(classes);

%% 1. Sample images from each class (training set)
fprintf('Generating sample images from each class...\n');

fig1 = figure('Position', [100, 100, 1200, 400]);
for classIdx = 1:numClasses
    % Find indices for this class
    indices = find(trainLabels == (classIdx - 1));

    % Randomly select 8 samples
    numSamples = min(8, length(indices));
    selectedIndices = indices(randperm(length(indices), numSamples));

    for i = 1:numSamples
        subplot(numClasses, numSamples, (classIdx-1)*numSamples + i);
        imshow(trainImages(:,:,selectedIndices(i)), []);
        if i == 1
            ylabel(sprintf('Class %s', classes{classIdx}), ...
                   'FontSize', 12, 'FontWeight', 'bold');
        end
        if classIdx == 1
            title(sprintf('Sample %d', i), 'FontSize', 10);
        end
    end
end

% No title - caption will be in LaTeX

% Save figure
saveas(fig1, fullfile(outputDir, 'class_samples.png'));
fprintf('  Saved: class_samples.png\n');

%% 2. Class distribution statistics (for reference, not plotted)
fprintf('Computing class distribution statistics...\n');

% Count samples per class
trainCounts = zeros(numClasses, 1);
testCounts = zeros(numClasses, 1);

for classIdx = 1:numClasses
    trainCounts(classIdx) = sum(trainLabels == (classIdx - 1));
    testCounts(classIdx) = sum(testLabels == (classIdx - 1));
end

fprintf('  Distribution computed (no figure generated - table exists in report)\n');

%% 3. Grid of random samples (Training)
fprintf('Generating training set sample grid...\n');

numSamplesPerClass = 4;
fig2 = figure('Position', [100, 100, 1000, 800]);

for classIdx = 1:numClasses
    indices = find(trainLabels == (classIdx - 1));
    selectedIndices = indices(randperm(length(indices), numSamplesPerClass));

    for i = 1:numSamplesPerClass
        subplot(numClasses, numSamplesPerClass, ...
                (classIdx-1)*numSamplesPerClass + i);
        imshow(trainImages(:,:,selectedIndices(i)), []);

        if i == 1
            ylabel(sprintf('%s', classes{classIdx}), ...
                   'FontSize', 14, 'FontWeight', 'bold');
        end

        if classIdx == 1
            title(sprintf('Sample %d', i), 'FontSize', 11);
        end

        axis off;
    end
end

% No title - caption will be in LaTeX

% Save figure
saveas(fig2, fullfile(outputDir, 'training_samples_grid.png'));
fprintf('  Saved: training_samples_grid.png\n');

%% 4. Grid of random samples (Validation)
fprintf('Generating validation set sample grid...\n');

numSamplesPerClass = 4;
fig3 = figure('Position', [100, 100, 1000, 800]);

for classIdx = 1:numClasses
    indices = find(testLabels == (classIdx - 1));
    selectedIndices = indices(randperm(length(indices), numSamplesPerClass));

    for i = 1:numSamplesPerClass
        subplot(numClasses, numSamplesPerClass, ...
                (classIdx-1)*numSamplesPerClass + i);
        imshow(testImages(:,:,selectedIndices(i)), []);

        if i == 1
            ylabel(sprintf('%s', classes{classIdx}), ...
                   'FontSize', 14, 'FontWeight', 'bold');
        end

        if classIdx == 1
            title(sprintf('Sample %d', i), 'FontSize', 11);
        end

        axis off;
    end
end

% No title - caption will be in LaTeX

% Save figure
saveas(fig3, fullfile(outputDir, 'validation_samples_grid.png'));
fprintf('  Saved: validation_samples_grid.png\n');

%% 5. Statistical summary
fprintf('\n=== Dataset Statistics ===\n');
fprintf('Training set:\n');
fprintf('  Total samples: %d\n', size(trainImages, 3));
fprintf('  Image size: %dx%d\n', size(trainImages, 1), size(trainImages, 2));
fprintf('  Classes: %d\n', numClasses);
fprintf('  Samples per class: %d\n', trainCounts(1));

fprintf('\nValidation set:\n');
fprintf('  Total samples: %d\n', size(testImages, 3));
fprintf('  Image size: %dx%d\n', size(testImages, 1), size(testImages, 2));
fprintf('  Classes: %d\n', numClasses);
fprintf('  Samples per class: %d\n', testCounts(1));

fprintf('\nClass distribution:\n');
for i = 1:numClasses
    fprintf('  Class %s: %d train, %d val (%.1f%% / %.1f%%)\n', ...
            classes{i}, trainCounts(i), testCounts(i), ...
            100*trainCounts(i)/sum(trainCounts), ...
            100*testCounts(i)/sum(testCounts));
end

fprintf('\nVisualization complete! Output saved to: %s\n', outputDir);

% Close all figures
close all;
