% Generate class_samples.png - Representative samples from each class
% Shows 8 random samples per class in a 7x8 grid

clear; close all; clc;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
src_dir = fileparts(script_dir);
project_root = fileparts(src_dir);
cd(project_root);

% Add paths
addpath(genpath(fullfile(src_dir, 'utils')));

fprintf('=== Generating class_samples.png ===\n\n');

%% Load data
fprintf('Loading training data...\n');
load('data/train.mat', 'data_train', 'labels_train');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);
samples_per_class = 8;

%% Select random samples for each class
fprintf('Selecting %d random samples per class...\n', samples_per_class);
selected_samples = cell(num_classes, 1);

for c = 0:num_classes-1
    % Find all samples of this class (labels are 0-indexed)
    class_idx = find(labels_train == c);

    % Randomly select 8 samples
    num_available = length(class_idx);
    if num_available >= samples_per_class
        selected_idx = randperm(num_available, samples_per_class);
    else
        % If fewer than 8, use all and pad with repetition
        selected_idx = 1:num_available;
        while length(selected_idx) < samples_per_class
            selected_idx = [selected_idx, selected_idx(1:min(samples_per_class-length(selected_idx), num_available))];
        end
        selected_idx = selected_idx(1:samples_per_class);
    end

    selected_samples{c+1} = class_idx(selected_idx);
end

%% Create figure
fprintf('Creating visualization...\n');
fig = figure('Position', [100, 100, 1400, 700], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
% Set tight spacing for subplots
set(fig, 'DefaultAxesLooseInset', [0.01, 0.01, 0.01, 0.01]);

% Create subplot grid: 8 rows total (1 header row + 7 class rows) x 9 columns
% Row 1: Header row for Sample labels
% Rows 2-8: Class rows (1 label column + 8 sample columns)

% First, create header row for Sample labels
for s = 1:samples_per_class
    subplot(num_classes + 1, samples_per_class + 1, s + 1);
    axis off;
    % Add "Sample X" label at the very top
    text(0.5, 0.5, sprintf('Sample %d', s), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'black');
end

% Leave first column of header row empty (aligns with class labels column)
subplot(num_classes + 1, samples_per_class + 1, 1);
axis off;

% Then create class rows
for c = 1:num_classes
    for s = 1:samples_per_class
        % Subplot index: (row+1-1)*num_cols + col (row+1 because header is row 1)
        % We use 9 columns: 1 label column + 8 sample columns
        subplot(num_classes + 1, samples_per_class + 1, (c)*(samples_per_class+1) + s + 1);

        sample_idx = selected_samples{c}(s);
        img = data_train(:, :, 1, sample_idx);

        imshow(img, []);
        axis off;
    end

    % Add class label on the left (first column of each row)
    subplot(num_classes + 1, samples_per_class + 1, (c)*(samples_per_class+1) + 1);
    axis off;
    % Create a larger text area for class label - vertical rotation
    text(0.5, 0.5, sprintf('Class %s', class_names{c}), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'Color', 'black', ...
        'Rotation', 90);  % Vertical text (rotated 90 degrees)
end

% Tight layout to minimize gaps - reduce spacing between subplots
% Reduce all subplot margins
for i = 1:(num_classes + 1) * (samples_per_class + 1)
    subplot(num_classes + 1, samples_per_class + 1, i);
    set(gca, 'LooseInset', [0, 0, 0, 0]);  % Remove all margins
    set(gca, 'Position', get(gca, 'OuterPosition'));  % Fill the subplot area
end

% Adjust figure size to be more compact
set(gcf, 'Units', 'normalized');
set(gcf, 'Position', [0.05, 0.05, 0.9, 0.85]);

% Save figure
output_path = fullfile('output', 'task7_1', 'class_samples.png');
if ~exist(fileparts(output_path), 'dir')
    mkdir(fileparts(output_path));
end

print(fig, output_path, '-dpng', '-r150');
fprintf('Saved: %s\n', output_path);

% Also copy to report directory
report_path = fullfile('..', 'ME5411-Project-Report', 'figs', 'task7_1', 'class_samples.png');
if exist(fileparts(report_path), 'dir')
    copyfile(output_path, report_path);
    fprintf('Copied to: %s\n', report_path);
end

close(fig);

fprintf('\n=== Complete ===\n');

