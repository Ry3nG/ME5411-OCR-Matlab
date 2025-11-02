% Generate All Final Figures for Report
% Consolidated script to generate only the figures we need

close all;
clear;

% Resolve project directories
script_dir = fileparts(mfilename('fullpath'));
src_dir = fileparts(script_dir);
project_root = fileparts(src_dir);

% Ensure the project root is the working directory for consistent paths
orig_dir = pwd;
cleanup = onCleanup(@() cd(orig_dir));
cd(project_root);

% Add required paths
addpath(fullfile(src_dir, 'core', 'network'));
addpath(fullfile(src_dir, 'utils'));

fprintf('=== Generating All Final Report Figures ===\n\n');

%% Configuration
log_dir = fullfile(project_root, 'output', 'task7_1', '11-02_21-09-20');
output_dir = fullfile(project_root, 'output', 'final_figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load model and data
fprintf('Loading model and data...\n');
load(fullfile(log_dir, 'cnn_best_acc.mat'), 'cnn');
load(fullfile(project_root, 'data', 'train.mat'), 'data_train', 'labels_train');
load(fullfile(project_root, 'data', 'test.mat'), 'data_test', 'labels_test');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% 1. Training curves from log
fprintf('\n1. Generating training curves...\n');
log_file = fullfile(project_root, 'output', 'logs', 'task7_1_training.log');
fid = fopen(log_file, 'r');
train_acc_history = [];
test_acc_history = [];

if fid ~= -1
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Epoch')
            tokens = regexp(line, 'Epoch (\d+): acc_test:([\d.]+) acc_train:([\d.]+)', 'tokens');
            if ~isempty(tokens)
                test_acc_history(end+1) = str2double(tokens{1}{2}); %#ok<AGROW>
                train_acc_history(end+1) = str2double(tokens{1}{3}); %#ok<AGROW>
            end
        end
    end
    fclose(fid);
else
    warning('Could not open training log: %s', log_file);
end

% Accuracy curve
fig = figure('Position', [100, 100, 800, 500]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
plot(1:length(train_acc_history), train_acc_history * 100, 'b-', 'LineWidth', 2);
hold on;
plot(1:length(test_acc_history), test_acc_history * 100, 'r-', 'LineWidth', 2);
grid on;
xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
legend('Training', 'Validation', 'Location', 'southeast');
ylim([0, 100]);
saveas(fig, fullfile(output_dir, 'accuracy_curve.png'));
fprintf('   Saved: accuracy_curve.png\n');
close(fig);

%% 2. Make predictions and compute confusion matrix
fprintf('\n2. Computing predictions and confusion matrix...\n');
cnn.train_mode = false;
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;  % Convert to 0-indexed

confusion_mat = zeros(num_classes, num_classes);
for i = 1:length(preds)
    true_label = labels_test(i) + 1;
    pred_label = preds(i) + 1;
    confusion_mat(true_label, pred_label) = confusion_mat(true_label, pred_label) + 1;
end

confusion_mat_norm = confusion_mat ./ sum(confusion_mat, 2);

fig = figure('Position', [100, 100, 700, 600]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(confusion_mat_norm);
colormap(flipud(gray));
colorbar;
caxis([0, 1]);

for i = 1:num_classes
    for j = 1:num_classes
        count = confusion_mat(i, j);
        percentage = confusion_mat_norm(i, j) * 100;
        if percentage > 50
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', text_color, 'FontSize', 10);
    end
end

xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
xticks(1:num_classes);
yticks(1:num_classes);
xticklabels(class_names);
yticklabels(class_names);
axis square;
saveas(fig, fullfile(output_dir, 'confusion_matrix.png'));
fprintf('   Saved: confusion_matrix.png\n');
close(fig);

%% 3. Misclassification examples (diverse, no repetition)
fprintf('\n3. Generating misclassification examples...\n');
misclassified_idx = find(preds ~= labels_test);

% Collect unique true/pred pairs
pairs = struct('key', {}, 'true', {}, 'pred', {}, 'idx', {});
for i = 1:length(misclassified_idx)
    idx = misclassified_idx(i);
    t = labels_test(idx);
    p = preds(idx);
    key = sprintf('%d_%d', t, p);

    found = false;
    for j = 1:length(pairs)
        if strcmp(pairs(j).key, key)
            found = true;
            break;
        end
    end

    if ~found
        pairs(end+1).key = key; %#ok<AGROW>
        pairs(end).true = t;
        pairs(end).pred = p;
        pairs(end).idx = idx;
    end
end

num_show = min(12, length(pairs));
fig = figure('Position', [100, 100, 1200, 300 * ceil(num_show/4)]);
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

for i = 1:num_show
    subplot(ceil(num_show/4), 4, i);
    imshow(data_test(:, :, 1, pairs(i).idx), []);
    title(sprintf('%s \\rightarrow %s', ...
        class_names{pairs(i).true+1}, class_names{pairs(i).pred+1}), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');
    axis off;
end

saveas(fig, fullfile(output_dir, 'misclassification_examples.png'));
fprintf('   Saved: misclassification_examples.png\n');
close(fig);

%% 4. Feature maps (individual files)
fprintf('\n4. Generating feature map images...\n');
fm_dir = fullfile(output_dir, 'feature_maps');
if ~exist(fm_dir, 'dir')
    mkdir(fm_dir);
end

% Select examples
d_to_4 = misclassified_idx(labels_test(misclassified_idx) == 5 & preds(misclassified_idx) == 1);
eight_to_a = misclassified_idx(labels_test(misclassified_idx) == 3 & preds(misclassified_idx) == 4);
d_correct = find(preds == labels_test & labels_test == 5);
zero_correct = find(preds == labels_test & labels_test == 0);

samples = [];
if ~isempty(d_to_4), samples(end+1).idx = d_to_4(1); samples(end).name = 'misclass_D_to_4'; end %#ok<AGROW>
if ~isempty(eight_to_a), samples(end+1).idx = eight_to_a(1); samples(end).name = 'misclass_8_to_A'; end %#ok<AGROW>
if ~isempty(d_correct), samples(end+1).idx = d_correct(1); samples(end).name = 'correct_D'; end %#ok<AGROW>
if ~isempty(zero_correct), samples(end+1).idx = zero_correct(1); samples(end).name = 'correct_0'; end %#ok<AGROW>

for s = 1:length(samples)
    img = data_test(:, :, 1, samples(s).idx);

    % Forward pass
    cnn.train_mode = false;
    cnn = forward(cnn, img, struct('train_mode', false));

    c1 = cnn.layers{2}.activations;
    c3 = cnn.layers{3}.activations;
    c5 = cnn.layers{4}.activations;

    % Input
    fig = figure('Position', [100, 100, 200, 200]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    imshow(img, []);
    axis off;
    saveas(fig, fullfile(fm_dir, [samples(s).name '_input.png']));
    close(fig);

    % C1: 2x2
    fig = figure('Position', [100, 100, 400, 400]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    for f = 1:4
        subplot(2, 2, f);
        imagesc(c1(:, :, f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fm_dir, [samples(s).name '_c1.png']));
    close(fig);

    % C3: 2x4
    fig = figure('Position', [100, 100, 800, 400]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    for f = 1:8
        subplot(2, 4, f);
        imagesc(c3(:, :, f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fm_dir, [samples(s).name '_c3.png']));
    close(fig);

    % C5: 4x4
    fig = figure('Position', [100, 100, 600, 600]);
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    for f = 1:16
        subplot(4, 4, f);
        imagesc(c5(:, :, f));
        colormap(gray);
        axis off;
    end
    saveas(fig, fullfile(fm_dir, [samples(s).name '_c5.png']));
    close(fig);

    fprintf('   Saved: %s (4 files)\n', samples(s).name);
end

fprintf('\n=== Generation Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('\nFigures for report:\n');
fprintf('  - accuracy_curve.png\n');
fprintf('  - confusion_matrix.png\n');
fprintf('  - misclassification_examples.png\n');
fprintf('  - feature_maps/ (individual feature visualisations)\n');
