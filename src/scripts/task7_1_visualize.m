% Task 7.1 Visualization
% Generate plots for CNN training results

clear all; %#ok<CLALL>
close all;

% Get the project root directory
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src/core'));
addpath(genpath('src/utils'));

% Load results from the latest run
result_dir = 'output/task7_1/11-03_03-17-04/';
output_dir = fullfile(result_dir, 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Loading results from: %s\n', result_dir);

% Load data
load(fullfile(result_dir, 'cnn_best_acc.mat'), 'cnn');
load(fullfile(result_dir, 'predictions.mat'), 'preds', 'labels_test_eval');
load(fullfile(result_dir, 'options.mat'), 'options');
load(fullfile(result_dir, 'acc_train.mat'), 'acc_train');
load(fullfile(result_dir, 'acc_test.mat'), 'acc_test');
load(fullfile(result_dir, 'loss_ar.mat'), 'loss_ar');
load('data/test.mat', 'data_test', 'labels_test');

% Convert labels
labels_test = labels_test + 1;

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% 1. Training Curves
fprintf('Plotting training curves...\n');
fig = figure('Position', [100, 100, 1200, 400], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');

% Accuracy curve
subplot(1, 3, 1);
epochs = 1:length(acc_train);
plot(epochs, acc_train * 100, 'b-', 'LineWidth', 2); hold on;
plot(epochs, acc_test * 100, 'r-', 'LineWidth', 2);
set(gca, 'Color', 'white');
set(gca, 'XColor', 'black');
set(gca, 'YColor', 'black');
xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
title('Training and Test Accuracy', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
leg = legend({'Train', 'Test'}, 'Location', 'southeast', 'TextColor', 'black', 'EdgeColor', 'black');
set(leg, 'Color', 'white');  % Set legend background to white
grid on;
set(gca, 'GridColor', [0.8, 0.8, 0.8]);
ylim([80 102]);

% Loss curve
subplot(1, 3, 2);
% Check if loss_ar exists and has data
if exist('loss_ar', 'var') && ~isempty(loss_ar) && length(loss_ar) > 0
    plot(1:length(loss_ar), loss_ar, 'k-', 'LineWidth', 1.5);
    set(gca, 'Color', 'white');
    set(gca, 'XColor', 'black');
    set(gca, 'YColor', 'black');
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    title('Training Loss', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);
else
    warning('loss_ar is empty or not found, parsing from training log...');
    % Try to parse from training_log.txt as fallback
    log_file = fullfile(result_dir, 'training_log.txt');
    if exist(log_file, 'file')
        loss_ar = [];
        fid = fopen(log_file, 'r');
        if fid ~= -1
            while ~feof(fid)
                line = fgetl(fid);
                if contains(line, 'loss:') && contains(line, 'iter')
                    tokens = regexp(line, 'loss:\s*([\d.]+)', 'tokens');
                    if ~isempty(tokens)
                        loss_ar = [loss_ar; str2double(tokens{1}{1})];
                    end
                end
            end
            fclose(fid);
        end
        if ~isempty(loss_ar)
            plot(1:length(loss_ar), loss_ar, 'k-', 'LineWidth', 1.5);
            set(gca, 'Color', 'white');
            set(gca, 'XColor', 'black');
            set(gca, 'YColor', 'black');
            xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
            ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
            title('Training Loss', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
            grid on;
            set(gca, 'GridColor', [0.8, 0.8, 0.8]);
        else
            warning('Could not parse loss data from log file');
        end
    end
end

% Learning rate schedule
subplot(1, 3, 3);
if ~exist('lr_ar', 'var')
    load(fullfile(result_dir, 'lr_ar.mat'), 'lr_ar');
end
if ~isempty(lr_ar)
    plot(1:length(lr_ar), lr_ar, 'g-', 'LineWidth', 2);
    set(gca, 'Color', 'white');
    set(gca, 'XColor', 'black');
    set(gca, 'YColor', 'black');
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');  % lr_ar is saved per epoch, not per iteration
    ylabel('Learning Rate', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    title('Learning Rate Schedule', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    grid on;
    set(gca, 'GridColor', [0.8, 0.8, 0.8]);
    % Note: Not using log scale since we have linear decay (0.1 -> 1e-5)
    % Linear on linear scale shows the true decay pattern
else
    warning('lr_ar is empty');
end

% Save figure with white background
set(fig, 'Color', 'white');
saveas(fig, fullfile(output_dir, 'training_curves.png'));
fprintf('Saved: training_curves.png\n');
close(fig);

%% 2. Confusion Matrix
fprintf('Computing confusion matrix...\n');
% Predict on test set
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;
labels_test_eval = labels_test - 1;

confMat = zeros(num_classes);
for i = 1:length(preds)
    true_label = labels_test_eval(i) + 1;
    pred_label = preds(i) + 1;
    confMat(true_label, pred_label) = confMat(true_label, pred_label) + 1;
end

% Plot confusion matrix (normalized, matching task7_2 style)
confMat_norm = confMat ./ sum(confMat, 2);

fig = figure('Position', [100, 100, 900, 800], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
imagesc(confMat_norm);
colormap(flipud(gray));  % Black for high values, white for low (matching task7_2)
colorbar;
caxis([0, 1]);
set(gca, 'Color', 'white');
set(gca, 'XColor', 'black');
set(gca, 'YColor', 'black');
xlabel('Predicted Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('True Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
set(gca, 'XTick', 1:num_classes, 'XTickLabel', class_names);
set(gca, 'YTick', 1:num_classes, 'YTickLabel', class_names);
axis square;
grid off;

% Add text annotations with count + percentage (matching task7_2)
for i = 1:num_classes
    for j = 1:num_classes
        count = confMat(i, j);
        percentage = confMat_norm(i, j) * 100;

        % Choose text color based on background intensity
        if confMat_norm(i, j) > 0.5
            textColor = 'white';
        else
            textColor = 'black';
        end

        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', textColor, 'FontSize', 11);
    end
end

saveas(fig, fullfile(output_dir, 'confusion_matrix.png'));
fprintf('Saved: confusion_matrix.png\n');
close(fig);

%% 3. Per-class Accuracy Bar Chart
fprintf('Plotting per-class accuracy...\n');
class_acc = zeros(num_classes, 1);
for i = 0:num_classes-1
    idx = (labels_test_eval == i);
    class_acc(i+1) = sum(preds(idx) == labels_test_eval(idx)) / sum(idx) * 100;
end

fig = figure('Position', [100, 100, 600, 400], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
bar(1:num_classes, class_acc, 'FaceColor', [0.2, 0.4, 0.6]);
set(gca, 'Color', 'white');
set(gca, 'XColor', 'black');
set(gca, 'YColor', 'black');
xlabel('Class', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
title('Per-Class Test Accuracy', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
set(gca, 'XTickLabel', class_names);
ylim([0 105]);
grid on;
set(gca, 'GridColor', [0.8, 0.8, 0.8]);

% Add value labels on bars
for i = 1:num_classes
    text(i, class_acc(i) + 1, sprintf('%.1f%%', class_acc(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'black');
end

saveas(fig, fullfile(output_dir, 'per_class_accuracy.png'));
fprintf('Saved: per_class_accuracy.png\n');
close(fig);

%% 4. Sample Predictions (Correct and Incorrect)
fprintf('Plotting sample predictions...\n');

% Find correctly classified samples (1 per class, single row)
% Strategy: Select "challenging" examples - correct but with lower confidence
% This demonstrates model robustness on non-trivial cases
fig = figure('Position', [100, 100, 1400, 200], 'Color', 'white');
set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
for c = 0:num_classes-1
    idx = find((labels_test_eval == c) & (preds == c));
    if length(idx) >= 1
        % Get prediction confidences for this class
        class_samples = data_test(:,:,1,idx);
        [pred_probs, ~] = predict(cnn, class_samples);

        % Find the correctly classified sample with MEDIAN confidence
        % (not highest - those are too easy, not lowest - might look wrong)
        % This gives us "interesting but still correct" examples
        confidences = zeros(length(idx), 1);
        for i = 1:length(idx)
            sample_pred = predict(cnn, data_test(:,:,1,idx(i)));
            % Get confidence for the true class (c+1 in 1-indexed)
            confidences(i) = cnn.layers{end}.activations(c+1);
        end

        % Sort by confidence and pick the one around 30-50th percentile
        % (challenging but not too close to decision boundary)
        [~, sorted_idx] = sort(confidences);
        select_position = max(1, round(length(sorted_idx) * 0.35)); % 35th percentile
        selected_idx = idx(sorted_idx(select_position));

        subplot(1, 7, c+1);
        imshow(data_test(:,:,1,selected_idx), []);
        title(sprintf('Class %s', class_names{c+1}), 'FontSize', 10, 'Color', 'black');
        axis off;
    end
end
saveas(fig, fullfile(output_dir, 'correct_predictions.png'));
fprintf('Saved: correct_predictions.png\n');
close(fig);

% Find misclassified samples (1 per class, single row)
% Note: Class '0' may have 0 misclassifications, so we may have fewer than 7
misclass_idx = find(preds ~= labels_test_eval);
if length(misclass_idx) > 0
    fig = figure('Position', [100, 100, 1400, 200], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    plot_idx = 1;
    for c = 0:num_classes-1
        % Find one misclassified sample from this true class
        idx = find((labels_test_eval == c) & (preds ~= c));
        if length(idx) >= 1
            subplot(1, 7, plot_idx);
            sample_idx = idx(1);
            imshow(data_test(:,:,1,sample_idx), []);
            true_class = class_names{labels_test_eval(sample_idx) + 1};
            pred_class = class_names{preds(sample_idx) + 1};
            title(sprintf('%s→%s', true_class, pred_class), ...
                'FontSize', 10, 'Color', 'red');
            axis off;
            plot_idx = plot_idx + 1;
        end
    end
    saveas(fig, fullfile(output_dir, 'misclassifications.png'));
    fprintf('Saved: misclassifications.png\n');
    close(fig);
end

%% 5. Feature Map Visualization
fprintf('Visualizing feature maps...\n');

% Select one correctly classified sample
correct_idx = find((labels_test_eval == 3) & (preds == 3), 1);  % Class 8
if ~isempty(correct_idx)
    sample_img = data_test(:,:,1,correct_idx);

    % Forward pass to get activations
    cnn_temp = cnn;
    cnn_temp.layers{end}.softmax = 0;  % Disable softmax for visualization
    local_option.train_mode = false;
    cnn_temp = forward(cnn_temp, sample_img, local_option);

    % Visualize first conv layer feature maps
    conv1_features = cnn_temp.layers{2}.activations;  % After first conv layer
    num_filters = size(conv1_features, 3);

    fig = figure('Position', [100, 100, 1000, 600], 'Color', 'white');
    set(fig, 'ToolBar', 'none', 'MenuBar', 'none');
    % Show original image
    subplot(4, 5, 1);
    imshow(sample_img, []);
    title('Input (8)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'black');
    axis off;

    % Show first 19 feature maps
    for i = 1:min(19, num_filters)
        subplot(4, 5, i+1);
        imshow(conv1_features(:,:,i), []);
        title(sprintf('Filter %d', i), 'FontSize', 8, 'Color', 'black');
        axis off;
    end
    sgtitle('Conv1 Feature Maps (16 filters, 5x5 kernel)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    saveas(fig, fullfile(output_dir, 'feature_maps_conv1.png'));
    fprintf('Saved: feature_maps_conv1.png\n');
    close(fig);
end

%% Summary
fprintf('\n=== Visualization Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('Generated figures:\n');
fprintf('  1. training_curves.png\n');
fprintf('  2. confusion_matrix.png\n');
fprintf('  3. per_class_accuracy.png\n');
fprintf('  4. correct_predictions.png\n');
fprintf('  5. misclassifications.png\n');
fprintf('  6. feature_maps_conv1.png\n');
